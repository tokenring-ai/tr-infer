//! Function calling in the model's own format. The template renders tools as JSON inside a
//! `<tools>` block and tool calls as
//! `<tool_call>\n<function=NAME>\n<parameter=K>\nVALUE\n</parameter>\n</function>\n</tool_call>`;
//! this module renders JSON the way the template's `tojson` does (Python `json.dumps` defaults,
//! as in transformers) and parses the model's calls back into OpenAI `tool_calls`, typing each
//! parameter by the tool's JSON schema (a non-string parameter is parsed as JSON).

use serde_json::{Map, Value};
use std::collections::HashMap;

/// `json.dumps(v, ensure_ascii=False)`: `", "` and `": "` separators, keys in the given order.
pub fn py_json(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => py_str(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_json(x, out);
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_str(k, out);
                out.push_str(": ");
                py_json(x, out);
            }
            out.push('}');
        }
    }
}

fn py_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// What the parser needs from a tool definition: the parameter types.
#[derive(Clone, Debug, Default)]
pub struct ToolSpec {
    pub name: String,
    pub param_types: HashMap<String, String>,
}

impl ToolSpec {
    /// From an OpenAI tool object (`{"type":"function","function":{...}}`) or a bare function.
    pub fn from_value(v: &Value) -> Option<ToolSpec> {
        let f = v.get("function").unwrap_or(v);
        let name = f.get("name")?.as_str()?.to_string();
        let mut param_types = HashMap::new();
        if let Some(props) = f.get("parameters").and_then(|p| p.get("properties")).and_then(|p| p.as_object()) {
            for (k, p) in props {
                let ty = match p.get("type") {
                    Some(Value::String(t)) => t.clone(),
                    Some(Value::Array(ts)) => ts.iter().filter_map(|t| t.as_str()).find(|t| *t != "null").unwrap_or("string").to_string(),
                    _ => String::new(), // no type (anyOf etc.): decided by the value
                };
                param_types.insert(k.clone(), ty);
            }
        }
        Some(ToolSpec { name, param_types })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Map<String, Value>,
}

fn type_value(raw: &str, ty: Option<&str>) -> Value {
    match ty {
        Some("string") => Value::String(raw.to_string()),
        Some(t) if !t.is_empty() => serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string())),
        // untyped: JSON if it is not a bare string, else the text
        _ => match serde_json::from_str::<Value>(raw) {
            Ok(Value::String(_)) | Err(_) => Value::String(raw.to_string()),
            Ok(v) => v,
        },
    }
}

/// Parse the inside of one `<tool_call>…</tool_call>` block.
pub fn parse_call(block: &str, specs: &[ToolSpec]) -> Option<ToolCall> {
    let s = block.trim();
    let rest = s.strip_prefix("<function=")?;
    let close = rest.find('>')?;
    let name = rest[..close].trim().to_string();
    let mut rest = &rest[close + 1..];
    let spec = specs.iter().find(|t| t.name == name);
    let mut arguments = Map::new();
    loop {
        let r = rest.trim_start();
        if r.starts_with("</function>") || r.is_empty() {
            break;
        }
        let r = r.strip_prefix("<parameter=")?;
        let close = r.find('>')?;
        let key = r[..close].trim().to_string();
        let mut val = &r[close + 1..];
        let end = val.find("</parameter>")?;
        rest = &val[end + "</parameter>".len()..];
        val = &val[..end];
        let val = val.strip_prefix('\n').unwrap_or(val);
        let val = val.strip_suffix('\n').unwrap_or(val);
        let ty = spec.and_then(|s| s.param_types.get(&key)).map(|t| t.as_str());
        arguments.insert(key, type_value(val, ty));
    }
    Some(ToolCall { name, arguments })
}

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

/// Splits a content stream into text and tool calls; holds back a possible partial `<tool_call>`.
pub struct ToolStream {
    buf: String,
    in_call: bool,
    specs: Vec<ToolSpec>,
    pub n_calls: usize,
}

pub enum Out {
    Text(String),
    Call(ToolCall),
}

impl ToolStream {
    pub fn new(specs: Vec<ToolSpec>) -> ToolStream {
        ToolStream { buf: String::new(), in_call: false, specs, n_calls: 0 }
    }

    pub fn push(&mut self, s: &str) -> Vec<Out> {
        self.buf.push_str(s);
        let mut out = Vec::new();
        loop {
            if self.in_call {
                match self.buf.find(CLOSE) {
                    Some(p) => {
                        let block = self.buf[..p].to_string();
                        self.buf.drain(..p + CLOSE.len());
                        self.in_call = false;
                        match parse_call(&block, &self.specs) {
                            Some(c) => {
                                self.n_calls += 1;
                                out.push(Out::Call(c));
                            }
                            None => out.push(Out::Text(format!("{OPEN}{block}{CLOSE}"))),
                        }
                    }
                    None => break,
                }
            } else if let Some(p) = self.buf.find(OPEN) {
                let text = self.buf[..p].to_string();
                self.buf.drain(..p + OPEN.len());
                self.in_call = true;
                // whitespace between the text and the call is formatting
                let text = if self.n_calls == 0 { text.trim_end().to_string() } else { text.trim().to_string() };
                if !text.is_empty() {
                    out.push(Out::Text(text));
                }
            } else {
                // hold back a partial "<tool_call>" and trailing whitespace that may precede one
                let mut hold = 0;
                for n in (1..OPEN.len()).rev() {
                    if self.buf.len() >= n && self.buf.is_char_boundary(self.buf.len() - n) && self.buf.ends_with(&OPEN[..n]) {
                        hold = n;
                        break;
                    }
                }
                let head = &self.buf[..self.buf.len() - hold];
                hold += head.len() - head.trim_end().len();
                let end = self.buf.len() - hold;
                if end > 0 {
                    out.push(Out::Text(self.buf[..end].to_string()));
                    self.buf.drain(..end);
                }
                break;
            }
        }
        out
    }

    /// End of generation: whatever is buffered is text (an unterminated call is returned raw).
    pub fn finish(&mut self) -> Vec<Out> {
        let mut out = Vec::new();
        let rest = std::mem::take(&mut self.buf);
        let rest = if self.in_call { format!("{OPEN}{rest}") } else { rest };
        self.in_call = false;
        let rest = if self.n_calls > 0 { rest.trim().to_string() } else { rest };
        if !rest.is_empty() {
            out.push(Out::Text(rest));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn python_json() {
        let v = json!({"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}});
        let mut s = String::new();
        py_json(&v, &mut s);
        assert_eq!(s, r#"{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}"#);
        let mut s = String::new();
        py_json(&json!(["a\"b\n", 1.5, true, null, "é<>"]), &mut s);
        assert_eq!(s, "[\"a\\\"b\\n\", 1.5, true, null, \"é<>\"]");
    }

    #[test]
    fn parse_and_type() {
        let spec = ToolSpec::from_value(&json!({"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},"units":{"type":["string","null"]},"opts":{"type":"object"}}}}})).unwrap();
        let c = parse_call("\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=opts>\n{\"a\": [1, 2]}\n</parameter>\n</function>\n", &[spec.clone()]).unwrap();
        assert_eq!(c.name, "get_weather");
        assert_eq!(Value::Object(c.arguments), json!({"city": "Paris", "days": 3, "opts": {"a": [1, 2]}}));
        // string-typed value that looks like JSON stays a string; untyped decides by the value
        let c = parse_call("<function=get_weather>\n<parameter=city>\n123\n</parameter>\n<parameter=extra>\n[1]\n</parameter>\n<parameter=note>\nhi\n</parameter>\n</function>", &[spec]).unwrap();
        assert_eq!(Value::Object(c.arguments), json!({"city": "123", "extra": [1], "note": "hi"}));
        // no parameters, multi-line value
        let c = parse_call("<function=f>\n<parameter=text>\nline1\nline2\n</parameter>\n</function>", &[]).unwrap();
        assert_eq!(c.arguments["text"], json!("line1\nline2"));
        assert!(parse_call("garbage", &[]).is_none());
    }

    #[test]
    fn stream_split() {
        let mut ts = ToolStream::new(vec![]);
        let mut text = String::new();
        let mut calls = Vec::new();
        let full = "Let me check.\n\n<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n</function>\n</tool_call>";
        // feed in awkward pieces
        for piece in ["Let me ", "check.\n\n<tool", "_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call", ">\n<tool_call>\n<function=b>\n</function>\n</tool_call>"] {
            for o in ts.push(piece) {
                match o {
                    Out::Text(t) => text.push_str(&t),
                    Out::Call(c) => calls.push(c),
                }
            }
        }
        for o in ts.finish() {
            match o {
                Out::Text(t) => text.push_str(&t),
                Out::Call(c) => calls.push(c),
            }
        }
        assert_eq!(text, "Let me check.");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[0].arguments["x"], json!(1));
        assert_eq!(calls[1].name, "b");
        assert!(calls[1].arguments.is_empty());
        let _ = full;
        // plain text with a lone '<' is passed through, held only while it could start a tag
        let mut ts = ToolStream::new(vec![]);
        let mut text = String::new();
        for piece in ["a < b <to", "ol x"] {
            for o in ts.push(piece) {
                if let Out::Text(t) = o {
                    text.push_str(&t);
                }
            }
        }
        for o in ts.finish() {
            if let Out::Text(t) = o {
                text.push_str(&t);
            }
        }
        assert_eq!(text, "a < b <tool x");
    }
}
