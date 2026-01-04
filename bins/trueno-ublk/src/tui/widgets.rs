//! Custom TUI widgets for trueno-ublk dashboard

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::Widget,
};

/// Horizontal bar chart for entropy distribution
pub struct EntropyBar {
    gpu_pct: u16,
    simd_pct: u16,
    scalar_pct: u16,
}

impl EntropyBar {
    pub fn new(gpu: u64, simd: u64, scalar: u64) -> Self {
        let total = gpu + simd + scalar;
        if total == 0 {
            return Self {
                gpu_pct: 0,
                simd_pct: 0,
                scalar_pct: 0,
            };
        }

        Self {
            gpu_pct: ((gpu * 100) / total) as u16,
            simd_pct: ((simd * 100) / total) as u16,
            scalar_pct: ((scalar * 100) / total) as u16,
        }
    }
}

impl Widget for EntropyBar {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let width = area.width as usize;
        let gpu_width = (width * self.gpu_pct as usize) / 100;
        let simd_width = (width * self.simd_pct as usize) / 100;
        let scalar_width = width.saturating_sub(gpu_width + simd_width);

        let mut x = area.x;

        // GPU portion (magenta)
        for _ in 0..gpu_width {
            if x < area.x + area.width {
                buf[(x, area.y)]
                    .set_char('█')
                    .set_style(Style::default().fg(Color::Magenta));
                x += 1;
            }
        }

        // SIMD portion (cyan)
        for _ in 0..simd_width {
            if x < area.x + area.width {
                buf[(x, area.y)]
                    .set_char('█')
                    .set_style(Style::default().fg(Color::Cyan));
                x += 1;
            }
        }

        // Scalar portion (yellow)
        for _ in 0..scalar_width {
            if x < area.x + area.width {
                buf[(x, area.y)]
                    .set_char('█')
                    .set_style(Style::default().fg(Color::Yellow));
                x += 1;
            }
        }
    }
}

/// Sparkline for throughput history
pub struct ThroughputSparkline<'a> {
    data: &'a [f64],
    max: f64,
}

impl<'a> ThroughputSparkline<'a> {
    pub fn new(data: &'a [f64]) -> Self {
        let max = data.iter().cloned().fold(0.0f64, f64::max);
        Self { data, max: max.max(1.0) }
    }
}

impl Widget for ThroughputSparkline<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        let width = area.width as usize;
        let data_start = self.data.len().saturating_sub(width);

        for (i, &value) in self.data[data_start..].iter().enumerate() {
            let x = area.x + i as u16;
            if x >= area.x + area.width {
                break;
            }

            let normalized = (value / self.max * 7.0).round() as usize;
            let bar_char = BARS[normalized.min(7)];

            buf[(x, area.y)]
                .set_char(bar_char)
                .set_style(Style::default().fg(Color::Green));
        }
    }
}

/// Compression ratio indicator
pub struct RatioIndicator {
    ratio: f64,
}

impl RatioIndicator {
    pub fn new(ratio: f64) -> Self {
        Self { ratio }
    }
}

impl Widget for RatioIndicator {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let color = if self.ratio >= 3.0 {
            Color::Green
        } else if self.ratio >= 2.0 {
            Color::Yellow
        } else if self.ratio >= 1.5 {
            Color::LightYellow
        } else {
            Color::Red
        };

        let text = format!("{:.2}:1", self.ratio);
        let x = area.x;
        for (i, ch) in text.chars().enumerate() {
            if x + (i as u16) < area.x + area.width {
                buf[(x + (i as u16), area.y)]
                    .set_char(ch)
                    .set_style(Style::default().fg(color));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_bar() {
        let bar = EntropyBar::new(50, 30, 20);
        assert_eq!(bar.gpu_pct, 50);
        assert_eq!(bar.simd_pct, 30);
        assert_eq!(bar.scalar_pct, 20);
    }

    #[test]
    fn test_entropy_bar_empty() {
        let bar = EntropyBar::new(0, 0, 0);
        assert_eq!(bar.gpu_pct, 0);
        assert_eq!(bar.simd_pct, 0);
        assert_eq!(bar.scalar_pct, 0);
    }

    #[test]
    fn test_ratio_indicator() {
        let ind = RatioIndicator::new(3.5);
        assert_eq!(ind.ratio, 3.5);
    }
}
