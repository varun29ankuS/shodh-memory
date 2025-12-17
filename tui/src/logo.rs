//! Shodh logo and branding ASCII art

/// Elephant logo - braille pixel art (6 lines, ~20 chars)
pub const ELEPHANT: &[&str] = &[
    "⠀⠀⠀⠀⠀⠀⠀⠀⣠⣤⣤⣤⣤⣄⠀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⢀⣤⣾⣿⣿⣿⣿⣿⣿⣿⣿⡀⠀⠀⠀⠀",
    "⠀⠀⠀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀",
    "⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⠈⠯⢹⣿⠀⠀⠀⠀",
    "⠀⠀⠀⣿⣿⣿⠟⠋⠉⠙⣿⣿⠀⠀⠀⠻⠷⠖⠀⠀",
    "⠀⠀⠐⠛⠛⠛⠀⠀⠀⠀⠛⠛⠃⠀⠀⠀⠀⠀⠀⠀",
];

/// Gradient colors for elephant (top to bottom: dark -> bright orange)
pub const ELEPHANT_GRADIENT: &[(u8, u8, u8)] = &[
    (150, 50, 20),  // Dark burnt orange
    (200, 70, 25),  // Deep orange
    (235, 90, 35),  // Bright orange
    (255, 110, 45), // Orange
    (255, 135, 60), // Light orange
    (255, 160, 85), // Pale orange/gold
];

/// SHODH text in big block letters
pub const SHODH_TEXT: &[&str] = &[
    "███████╗██╗  ██╗ ██████╗ ██████╗ ██╗  ██╗",
    "██╔════╝██║  ██║██╔═══██╗██╔══██╗██║  ██║",
    "███████╗███████║██║   ██║██║  ██║███████║",
    "╚════██║██╔══██║██║   ██║██║  ██║██╔══██║",
    "███████║██║  ██║╚██████╔╝██████╔╝██║  ██║",
    "╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝",
];

/// Gradient colors for SHODH text
pub const SHODH_GRADIENT: &[(u8, u8, u8)] = &[
    (255, 80, 30),  // Bright orange-red
    (255, 100, 40), // Orange
    (255, 120, 50), // Light orange
    (255, 100, 40), // Orange
    (255, 80, 30),  // Bright orange-red
    (200, 60, 20),  // Dark orange
];

/// Tagline
pub const TAGLINE: &str = "M E M O R Y   S Y S T E M";
