//! Role tags and the chunking of a token sequence along role boundaries.
use serde::{Deserialize, Serialize};

/// What the tokens of a chunk are: which side of the conversation wrote them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// System prompt, including the server-injected instructions and tool definitions.
    System,
    /// A user turn.
    User,
    /// The model's `<think>` block (in the prompt as `reasoning_content`, or generated).
    Reasoning,
    /// A tool result (`<tool_response>` block).
    Tool,
    /// The model's answer, including tool calls it emitted.
    Assistant,
}

impl Role {
    pub const ALL: [Role; 5] = [Role::System, Role::User, Role::Reasoning, Role::Tool, Role::Assistant];
    pub fn name(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Reasoning => "reasoning",
            Role::Tool => "tool",
            Role::Assistant => "assistant",
        }
    }
    pub fn parse(s: &str) -> Option<Role> {
        Some(match s {
            "system" => Role::System,
            "user" => Role::User,
            "reasoning" => Role::Reasoning,
            "tool" => Role::Tool,
            "assistant" => Role::Assistant,
            _ => return None,
        })
    }
}

/// A run of token positions `[start, end)` with one role.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub role: Role,
}

impl Span {
    pub fn new(start: usize, end: usize, role: Role) -> Span {
        Span { start, end, role }
    }
    pub fn len(&self) -> usize {
        self.end - self.start
    }
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

/// Cut `[from, ..)` of the spans into chunks: a chunk never crosses a role boundary and holds at
/// most `chunk_len` tokens. Spans must be sorted and non-overlapping.
pub fn chunk_spans(spans: &[Span], from: usize, chunk_len: usize) -> Vec<Span> {
    let chunk_len = chunk_len.max(1);
    let mut out = Vec::new();
    for s in spans {
        let a = s.start.max(from);
        if a >= s.end {
            continue;
        }
        let mut p = a;
        while p < s.end {
            let e = (p + chunk_len).min(s.end);
            out.push(Span::new(p, e, s.role));
            p = e;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_end_at_role_boundaries_then_every_chunk_len() {
        let spans = [Span::new(0, 10, Role::System), Span::new(10, 25, Role::User), Span::new(25, 27, Role::Reasoning)];
        let c = chunk_spans(&spans, 0, 8);
        assert_eq!(
            c,
            vec![Span::new(0, 8, Role::System), Span::new(8, 10, Role::System), Span::new(10, 18, Role::User), Span::new(18, 25, Role::User), Span::new(25, 27, Role::Reasoning)]
        );
        // `from` clips the first chunk
        let c = chunk_spans(&spans, 12, 100);
        assert_eq!(c, vec![Span::new(12, 25, Role::User), Span::new(25, 27, Role::Reasoning)]);
        assert!(chunk_spans(&spans, 27, 8).is_empty());
        assert_eq!(chunk_spans(&spans, 0, 0).len(), 27, "chunk_len 0 behaves as 1");
    }

    #[test]
    fn role_names_round_trip() {
        for r in [Role::System, Role::User, Role::Reasoning, Role::Tool, Role::Assistant] {
            assert_eq!(Role::parse(r.name()), Some(r));
            let j = serde_json::to_string(&r).unwrap();
            assert_eq!(j, format!("\"{}\"", r.name()));
        }
        assert_eq!(Role::parse("developer"), None);
    }
}
