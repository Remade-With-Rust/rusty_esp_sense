//! A capture: the rows of one recording, in the ledger's fixture format.
//!
//! `CSI_DATA,<rssi>,<len>,<i>,<q>,…` -- the format of every fixture under
//! `rusty_esp_signal/.../tests/fixtures/csi`, of the Cuenca dataset, and of
//! what `rusty_esp_iroh-host`'s reference client writes as `csi.csv` from a
//! live W5 stream. A row may end with one more field, the frame's time in
//! seconds since the capture began (the Cuenca dataset's fourth scenario
//! writes one); when it is there it stamps the frame, and when it is not
//! the frame is stamped at the rate the recording was made at. Anything
//! else after the I/Q refuses the row.

use crate::prof::{self, Counter, Stage};
use rusty_esp_signal_core::esp_core::Micros;
use rusty_esp_signal_core::radar::csi_stream::{MAX_IQ, Sample};

/// One recording.
#[derive(Debug, Clone)]
pub struct Capture {
    /// Where it came from (a file name).
    pub name: String,
    /// Every row that parsed, in order.
    pub samples: Vec<Sample>,
    /// Rows that did not: wrong tag, a length that does not match its
    /// values, a value that is not an `i8`.
    pub rejected: usize,
}

/// Parse a capture. `layout` is the `csi_stream` tag the buffers were
/// captured with (the file does not say); `frame_us` the interval to stamp
/// frames at (20 000 for 50 Hz).
#[must_use]
pub fn parse(name: &str, text: &str, layout: u8, frame_us: u64) -> Capture {
    // `text` is valid UTF-8 already, so the only error cannot occur.
    parse_bytes(name, text.as_bytes(), layout, frame_us).unwrap_or_else(|_| Capture {
        name: name.to_owned(),
        samples: Vec::new(),
        rejected: 0,
    })
}

/// The error `read_to_string` gives for bytes that are not UTF-8, which is
/// what reading a capture has always returned for one.
fn not_utf8() -> crate::Error {
    crate::Error::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "stream did not contain valid UTF-8",
    ))
}

/// [`parse`] over raw bytes, without validating the whole file as UTF-8
/// first.
///
/// A capture is ASCII, and an ASCII line is parsed by [`row_ascii`], which
/// accepts exactly what [`row`] accepts on it (an oracle test holds that).
/// A line with any other byte is validated and parsed by [`row`], so a
/// capture that is not UTF-8 is refused exactly as `read_to_string` refused
/// it: a UTF-8 sequence cannot contain a newline, so validating line by line
/// is validating the file.
///
/// # Errors
///
/// [`crate::Error::Io`] (`InvalidData`) when the bytes are not UTF-8.
pub fn parse_bytes(name: &str, bytes: &[u8], layout: u8, frame_us: u64) -> crate::Result<Capture> {
    let _g = prof::scope(Stage::Parse);
    let mut samples = Vec::with_capacity(bytes.len() / 400);
    let mut rejected = 0usize;
    for raw in bytes.split(|&b| b == b'\n') {
        let parsed = if raw.is_ascii() {
            let line = trim_ascii(raw);
            if line.is_empty() {
                continue;
            }
            row_ascii(line)
        } else {
            prof::add(Counter::Utf8Bytes, raw.len() as u64);
            let text = core::str::from_utf8(raw).map_err(|_| not_utf8())?;
            let line = text.trim();
            if line.is_empty() {
                continue;
            }
            row(line)
        };
        match parsed {
            Some((rssi, iq, n, t)) => {
                let at = match t {
                    Some(secs) => Micros((secs * 1e6).round() as u64),
                    None => Micros(frame_us * samples.len() as u64),
                };
                match Sample::from_iq(at, rssi, 0, layout, &iq[..n]) {
                    Ok(s) => samples.push(s),
                    Err(_) => rejected += 1,
                }
            }
            None => rejected += 1,
        }
    }
    prof::add(Counter::Rows, (samples.len() + rejected) as u64);
    Ok(Capture {
        name: name.to_owned(),
        samples,
        rejected,
    })
}

/// Read a capture from a file.
///
/// # Errors
///
/// [`crate::Error::Io`] when the file cannot be read or is not UTF-8.
pub fn read(path: &std::path::Path, layout: u8, frame_us: u64) -> crate::Result<Capture> {
    let bytes = {
        let _g = prof::scope(Stage::Parse);
        std::fs::read(path)?
    };
    let name = path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    );
    parse_bytes(&name, &bytes, layout, frame_us)
}

/// What `char::is_whitespace` (and so `str::trim`) calls whitespace, within
/// ASCII: tab, line feed, vertical tab, form feed, carriage return, space.
/// Not `u8::is_ascii_whitespace`, which leaves out the vertical tab.
const fn is_space(b: u8) -> bool {
    matches!(b, b'\t' | b'\n' | 0x0B | 0x0C | b'\r' | b' ')
}

fn trim_ascii(mut s: &[u8]) -> &[u8] {
    while let [first, rest @ ..] = s {
        if !is_space(*first) {
            break;
        }
        s = rest;
    }
    while let [rest @ .., last] = s {
        if !is_space(*last) {
            break;
        }
        s = rest;
    }
    s
}

/// `i8::from_str`, on ASCII: an optional sign, at least one digit, nothing
/// else, in range.
fn parse_i8(f: &[u8]) -> Option<i8> {
    let (neg, digits) = match f.first()? {
        b'-' => (true, &f[1..]),
        b'+' => (false, &f[1..]),
        _ => (false, f),
    };
    if digits.is_empty() {
        return None;
    }
    let mut v: i32 = 0;
    for &d in digits {
        if !d.is_ascii_digit() {
            return None;
        }
        v = v * 10 + i32::from(d - b'0');
        if v > 128 {
            return None;
        }
    }
    let v = if neg { -v } else { v };
    i8::try_from(v).ok()
}

/// `usize::from_str` capped at `MAX_IQ`, on ASCII: an optional `+`, at least
/// one digit, nothing else. Anything above `MAX_IQ` is refused, as the
/// caller refuses it.
fn parse_len(f: &[u8]) -> Option<usize> {
    let digits = match f.first()? {
        b'+' => &f[1..],
        _ => f,
    };
    if digits.is_empty() {
        return None;
    }
    let mut v = 0usize;
    for &d in digits {
        if !d.is_ascii_digit() {
            return None;
        }
        v = v * 10 + usize::from(d - b'0');
        if v > MAX_IQ {
            return None;
        }
    }
    Some(v)
}

/// [`row`], for an ASCII line.
fn row_ascii(line: &[u8]) -> Option<(i8, [i8; MAX_IQ], usize, Option<f64>)> {
    let mut fields = line.split(|&b| b == b',');
    if fields.next()? != b"CSI_DATA" {
        return None;
    }
    let rssi = parse_i8(trim_ascii(fields.next()?))?;
    let n = parse_len(trim_ascii(fields.next()?))?;
    let mut iq = [0i8; MAX_IQ];
    for slot in iq.iter_mut().take(n) {
        *slot = parse_i8(trim_ascii(fields.next()?))?;
    }
    let t = match fields.next() {
        None => None,
        Some(f) => {
            prof::add(Counter::FieldParses, 1);
            let secs: f64 = core::str::from_utf8(trim_ascii(f)).ok()?.parse().ok()?;
            if !secs.is_finite() || secs < 0.0 {
                return None;
            }
            Some(secs)
        }
    };
    if fields.next().is_some() {
        return None;
    }
    Some((rssi, iq, n, t))
}

fn row(line: &str) -> Option<(i8, [i8; MAX_IQ], usize, Option<f64>)> {
    let mut fields = line.split(',');
    if fields.next()? != "CSI_DATA" {
        return None;
    }
    prof::add(Counter::FieldParses, 2);
    let rssi: i8 = fields.next()?.trim().parse().ok()?;
    let n: usize = fields.next()?.trim().parse().ok()?;
    if n > MAX_IQ {
        return None;
    }
    let mut iq = [0i8; MAX_IQ];
    prof::add(Counter::FieldParses, n as u64);
    for slot in iq.iter_mut().take(n) {
        *slot = fields.next()?.trim().parse().ok()?;
    }
    let t = match fields.next() {
        None => None,
        Some(f) => {
            let secs: f64 = f.trim().parse().ok()?;
            if !secs.is_finite() || secs < 0.0 {
                return None;
            }
            Some(secs)
        }
    };
    if fields.next().is_some() {
        return None;
    }
    Some((rssi, iq, n, t))
}

#[cfg(test)]
mod tests {
    use rusty_esp_signal_core::radar::csi_stream::TAG_C6_HT20_NATURAL;

    use super::*;

    fn line(rssi: i8, v: i8) -> String {
        let mut s = format!("CSI_DATA,{rssi},128");
        for k in 0..128 {
            s.push_str(&format!(",{}", if k % 2 == 0 { v } else { 0 }));
        }
        s
    }

    #[test]
    fn rows_become_samples_stamped_at_the_frame_rate() {
        let text = format!("{}\n{}\n", line(-40, 10), line(-41, 11));
        let c = parse("x.csv", &text, TAG_C6_HT20_NATURAL, 20_000);
        assert_eq!(c.samples.len(), 2);
        assert_eq!(c.rejected, 0);
        assert_eq!(c.samples[1].at, Micros(20_000));
        assert_eq!(c.samples[1].rssi, -41);
        assert_eq!(c.samples[0].iq()[0], 10);
        assert_eq!(c.samples[0].layout, TAG_C6_HT20_NATURAL);
    }

    /// R6's oracle: the byte parser accepts exactly what the `str` parser
    /// accepts, field for field, over lines built from every case that
    /// matters -- signs, leading zeros, range edges, every whitespace
    /// `str::trim` removes, empty and extra fields, trailing times.
    #[test]
    fn the_byte_parser_accepts_exactly_what_the_str_parser_does() {
        let tokens = [
            "0", "7", "-7", "+7", "007", "-0", "+0", "127", "128", "-128", "-129", "300", "", "+",
            "-", "+-1", "1a", " 5", "5 ", "\t5", "\x0b5", "5\x0c", "\r5", " -12 ",
        ];
        let lens = ["4", "+4", "004", "3", "5", "0", "", "-4", "129", "4a"];
        let tails = [
            "", ",0.5", ",1e3", ",-1", ",nan", ",inf", ",0.5,9", ", 2.25 ", ",", ",x",
        ];
        let mut seed = 0x9E37_79B9_u64;
        let mut next = |m: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % m as u64) as usize
        };
        let mut checked = 0;
        for tag in ["CSI_DATA", "CSI_DAT", " CSI_DATA", "csi_data"] {
            for _ in 0..600 {
                let mut line = format!(
                    "{tag},{},{}",
                    tokens[next(tokens.len())],
                    lens[next(lens.len())]
                );
                for _ in 0..next(6) {
                    line.push(',');
                    line.push_str(tokens[next(tokens.len())]);
                }
                line.push_str(tails[next(tails.len())]);
                let trimmed = line.trim();
                let old = row(trimmed);
                let new = row_ascii(trim_ascii(line.as_bytes()));
                assert_eq!(
                    old.map(|r| (r.0, r.1, r.2, r.3.map(f64::to_bits))),
                    new.map(|r| (r.0, r.1, r.2, r.3.map(f64::to_bits))),
                    "{line:?}"
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 2400);
    }

    /// The whole-capture path: bytes and text parse to the same samples;
    /// a non-ASCII line goes the old way; bytes that are not UTF-8 are
    /// refused as reading them as text refused them.
    #[test]
    fn bytes_and_text_parse_alike_and_bad_utf8_is_refused() {
        let text = format!(
            "{}\r\n\n  {}\u{a0}\n{}\n",
            line(-40, 1),
            line(-41, 2),
            line(-42, 3)
        );
        let a = parse("x", &text, TAG_C6_HT20_NATURAL, 20_000);
        let b = parse_bytes("x", text.as_bytes(), TAG_C6_HT20_NATURAL, 20_000).unwrap();
        assert_eq!(a.samples, b.samples);
        assert_eq!(a.rejected, b.rejected);
        assert_eq!(a.samples.len(), 3, "the U+00A0 line trims like str::trim");
        let mut bad = line(-40, 1).into_bytes();
        bad.push(0xFF);
        assert!(parse_bytes("x", &bad, 0, 20_000).is_err());
    }

    #[test]
    fn a_trailing_time_stamps_the_frame() {
        let text = format!("{},0.023\n{},0.043\n", line(-40, 1), line(-40, 1));
        let c = parse("x.csv", &text, 0, 20_000);
        assert_eq!(c.rejected, 0);
        assert_eq!(c.samples[0].at, Micros(23_000));
        assert_eq!(c.samples[1].at, Micros(43_000));
    }

    #[test]
    fn a_row_that_lies_about_itself_is_counted_not_read() {
        let short = "CSI_DATA,-40,128,1,2,3";
        let long = format!("{},0.5,9", line(-40, 1));
        let bad = line(-40, 1).replacen(",1,", ",300,", 1);
        let text = format!("{short}\n{long}\n{bad}\nnot a row\n{}\n", line(-40, 1));
        let c = parse("x.csv", &text, 0, 20_000);
        assert_eq!(c.samples.len(), 1);
        assert_eq!(c.rejected, 4);
    }
}
