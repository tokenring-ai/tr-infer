//! Minimal HTTP/1.1 on std: one thread per connection, keep-alive, Content-Length and chunked
//! request bodies, `Expect: 100-continue`, chunked responses for server-sent events.
//! The engine serialises requests anyway, so an async stack buys nothing here.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpStream;

pub const MAX_HEADER: usize = 64 << 10;
pub const MAX_BODY: usize = 64 << 20;

pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>, // names lower-cased
    pub body: Vec<u8>,
    pub keep_alive: bool,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

/// Read one request from a keep-alive connection. `Ok(None)` on a clean close before any bytes.
pub fn read_request(r: &mut BufReader<TcpStream>) -> io::Result<Option<Request>> {
    let mut line = String::new();
    loop {
        line.clear();
        if r.read_line(&mut line)? == 0 {
            return Ok(None);
        }
        if line.len() > MAX_HEADER {
            return Err(bad("request line too long"));
        }
        if !line.trim_end_matches(['\r', '\n']).is_empty() {
            break;
        }
    }
    let (method, target, version) = {
        let mut parts = line.trim_end_matches(['\r', '\n']).split(' ');
        let m = parts.next().unwrap_or("").to_string();
        let t = parts.next().ok_or_else(|| bad("missing request target"))?.to_string();
        (m, t, parts.next().unwrap_or("HTTP/1.0").to_string())
    };
    let path = target.split('?').next().unwrap_or("").to_string();
    let mut headers = Vec::new();
    let mut total = line.len();
    loop {
        line.clear();
        if r.read_line(&mut line)? == 0 {
            return Err(bad("connection closed inside headers"));
        }
        total += line.len();
        if total > MAX_HEADER {
            return Err(bad("headers too large"));
        }
        let l = line.trim_end_matches(['\r', '\n']);
        if l.is_empty() {
            break;
        }
        let (k, v) = l.split_once(':').ok_or_else(|| bad("malformed header"))?;
        headers.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
    }
    let hdr = |n: &str| headers.iter().find(|(k, _)| k == n).map(|(_, v)| v.as_str());
    let conn = hdr("connection").map(|v| v.to_ascii_lowercase()).unwrap_or_default();
    let keep_alive = if version == "HTTP/1.0" { conn.contains("keep-alive") } else { !conn.contains("close") };
    let chunked = hdr("transfer-encoding").map(|v| v.to_ascii_lowercase().contains("chunked")).unwrap_or(false);
    let clen: usize = hdr("content-length").map(|v| v.parse().map_err(|_| bad("bad content-length"))).transpose()?.unwrap_or(0);
    if clen > MAX_BODY {
        return Err(bad("body too large"));
    }
    if (chunked || clen > 0) && hdr("expect").map(|v| v.eq_ignore_ascii_case("100-continue")).unwrap_or(false) {
        r.get_mut().write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
    }
    let mut body = Vec::new();
    if chunked {
        loop {
            line.clear();
            r.read_line(&mut line)?;
            let size = usize::from_str_radix(line.trim().split(';').next().unwrap_or("").trim(), 16).map_err(|_| bad("bad chunk size"))?;
            if size == 0 {
                // trailers until the empty line
                loop {
                    line.clear();
                    if r.read_line(&mut line)? == 0 || line.trim_end_matches(['\r', '\n']).is_empty() {
                        break;
                    }
                }
                break;
            }
            if body.len() + size > MAX_BODY {
                return Err(bad("body too large"));
            }
            let at = body.len();
            body.resize(at + size, 0);
            r.read_exact(&mut body[at..])?;
            let mut crlf = [0u8; 2];
            r.read_exact(&mut crlf)?;
        }
    } else if clen > 0 {
        body.resize(clen, 0);
        r.read_exact(&mut body)?;
    }
    Ok(Some(Request { method, path, headers, body, keep_alive }))
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Unknown",
    }
}

const CORS: &str = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Authorization, Content-Type\r\n";

pub fn write_response(w: &mut TcpStream, status: u16, ctype: &str, body: &[u8], keep_alive: bool) -> io::Result<()> {
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: {}\r\n{}\r\n",
        status,
        reason(status),
        ctype,
        body.len(),
        if keep_alive { "keep-alive" } else { "close" },
        CORS
    );
    w.write_all(head.as_bytes())?;
    w.write_all(body)?;
    w.flush()
}

/// Start a chunked `text/event-stream` response.
pub fn begin_event_stream(w: &mut TcpStream, keep_alive: bool) -> io::Result<()> {
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: {}\r\n{}\r\n",
        if keep_alive { "keep-alive" } else { "close" },
        CORS
    );
    w.write_all(head.as_bytes())?;
    w.flush()
}

pub fn write_chunk(w: &mut TcpStream, data: &[u8]) -> io::Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    w.write_all(format!("{:x}\r\n", data.len()).as_bytes())?;
    w.write_all(data)?;
    w.write_all(b"\r\n")?;
    w.flush()
}

pub fn end_stream(w: &mut TcpStream) -> io::Result<()> {
    w.write_all(b"0\r\n\r\n")?;
    w.flush()
}

/// True when the peer has closed its side (nothing more will be read from it).
pub fn peer_closed(s: &TcpStream) -> bool {
    if s.set_nonblocking(true).is_err() {
        return true;
    }
    let mut b = [0u8; 1];
    let closed = match s.peek(&mut b) {
        Ok(0) => true,
        Ok(_) => false,
        Err(e) if e.kind() == io::ErrorKind::WouldBlock => false,
        Err(_) => true,
    };
    let _ = s.set_nonblocking(false);
    closed
}
