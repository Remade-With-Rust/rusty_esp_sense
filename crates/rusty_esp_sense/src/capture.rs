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
    let _g = prof::scope(Stage::Parse);
    let mut samples = Vec::with_capacity(text.len() / 400);
    let mut rejected = 0usize;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match row(line) {
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
    Capture {
        name: name.to_owned(),
        samples,
        rejected,
    }
}

/// Read a capture from a file.
///
/// # Errors
///
/// [`crate::Error::Io`] when the file cannot be read.
pub fn read(path: &std::path::Path, layout: u8, frame_us: u64) -> crate::Result<Capture> {
    let text = {
        let _g = prof::scope(Stage::Parse);
        let t = std::fs::read_to_string(path)?;
        prof::add(Counter::Utf8Bytes, t.len() as u64);
        t
    };
    let name = path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    );
    Ok(parse(&name, &text, layout, frame_us))
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
