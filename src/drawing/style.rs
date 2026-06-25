use std::{
    borrow::Borrow,
    hash::Hash,
    sync::{Arc, LazyLock, RwLock, RwLockReadGuard},
};

use crate::util::HashMap;

static GLOBAL_THEME: LazyLock<Arc<RwLock<Theme>>> = LazyLock::new(|| Arc::new(RwLock::new(Theme::default())));

#[derive(Debug, Clone)]
pub struct Theme {
    pub propagators: HashMap<String, PathStyle>,
    pub vertices: HashMap<String, Decoration>,

    pub pattern_count: usize,
    pub decoration_size: f64,
    pub wavy_amplitude: f64,
    pub curly_amplitude: f64,
    pub self_loop_size: f64,
    pub font_size: f64,
    pub character_width: f64,
    pub propagator_label_distance: f64,

    pub border_size: f64,
    pub title_size: f64,

    pub self_loop_multiplier: usize,
    pub self_loop_reduction: f64,

    pub label_propagators: bool,
}

impl Theme {
    pub fn get_global() -> RwLockReadGuard<'static, Self> {
        match GLOBAL_THEME.read() {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }

    pub fn set_global(theme: Self) {
        match GLOBAL_THEME.write() {
            Ok(mut t) => *t = theme,
            Err(e) => panic!("{}", e),
        }
    }

    pub fn set_particle(&mut self, p: String, style: PathStyle) -> &mut Self {
        self.propagators.insert(p, style);
        self
    }

    pub(crate) fn get_particle<Q>(&self, p: &Q) -> Option<&PathStyle>
    where
        Q: Hash + Eq + ?Sized,
        String: Borrow<Q>,
    {
        self.propagators.get(p)
    }

    pub(crate) fn get_particle_or_default<Q>(&self, p: &Q) -> PathStyle
    where
        Q: Hash + Eq + ?Sized,
        String: Borrow<Q>,
    {
        match self.get_particle(p) {
            Some(style) => *style,
            None => PathStyle::default(),
        }
    }

    pub fn set_vertex(&mut self, v: String, style: Decoration) -> &mut Self {
        self.vertices.insert(v, style);
        self
    }

    pub fn get_vertex<Q>(&self, v: &Q) -> Option<&Decoration>
    where
        Q: Hash + Eq + ?Sized,
        String: Borrow<Q>,
    {
        self.vertices.get(v)
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            propagators: HashMap::default(),
            vertices: HashMap::default(),

            pattern_count: 8,
            decoration_size: 0.025,
            wavy_amplitude: 0.075,
            curly_amplitude: 0.055,
            self_loop_size: 0.3,
            font_size: 0.06,
            character_width: 0.02022,
            propagator_label_distance: 0.05,

            border_size: 0.05,
            title_size: 0.15,

            self_loop_multiplier: 3,
            self_loop_reduction: 0.5,

            label_propagators: true,
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub enum Stroke {
    #[default]
    Solid,
    Dashed,
    Dotted,
}

#[derive(Debug, Clone, Copy)]
pub struct PathStyle {
    pub stroke: Stroke,
    pub width: f64,
    pub color: Color,
    pub fill: bool,
    pub close: bool,
    pub decoration: Option<Decoration>,
    pub n_segments: usize,
}

impl PathStyle {
    pub fn stroke(mut self, stroke: Stroke) -> Self {
        self.stroke = stroke;
        self
    }
    pub fn width(mut self, width: f64) -> Self {
        self.width = width;
        self
    }
    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }
    pub fn fill(mut self, fill: bool) -> Self {
        self.fill = fill;
        self
    }
    pub fn close(mut self, close: bool) -> Self {
        self.close = close;
        self
    }
    pub fn decoration(mut self, decoration: Decoration) -> Self {
        self.decoration = Some(decoration);
        self
    }
    pub fn n_segments(mut self, n_segments: usize) -> Self {
        self.n_segments = n_segments;
        self
    }
}

impl Default for PathStyle {
    fn default() -> Self {
        Self {
            width: 0.006,
            n_segments: 8,
            stroke: Default::default(),
            color: Default::default(),
            fill: Default::default(),
            close: Default::default(),
            decoration: Default::default(),
        }
    }
}

#[derive(Default, Clone)]
pub enum Anchor {
    Left,
    #[default]
    Center,
    Right,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub const BLACK: Self = Self::from_rgb(0, 0, 0);
    pub const DARK_GRAY: Self = Self::from_rgb(96, 96, 96);
    pub const GRAY: Self = Self::from_rgb(160, 160, 160);
    pub const LIGHT_GRAY: Self = Self::from_rgb(220, 220, 220);
    pub const WHITE: Self = Self::from_rgb(255, 255, 255);

    pub const BROWN: Self = Self::from_rgb(165, 42, 42);
    pub const DARK_RED: Self = Self::from_rgb(0x8B, 0, 0);
    pub const RED: Self = Self::from_rgb(255, 0, 0);
    pub const LIGHT_RED: Self = Self::from_rgb(255, 128, 128);

    pub const CYAN: Self = Self::from_rgb(0, 255, 255);
    pub const MAGENTA: Self = Self::from_rgb(255, 0, 255);
    pub const YELLOW: Self = Self::from_rgb(255, 255, 0);

    pub const ORANGE: Self = Self::from_rgb(255, 165, 0);
    pub const LIGHT_YELLOW: Self = Self::from_rgb(255, 255, 0xE0);
    pub const KHAKI: Self = Self::from_rgb(240, 230, 140);

    pub const DARK_GREEN: Self = Self::from_rgb(0, 0x64, 0);
    pub const GREEN: Self = Self::from_rgb(0, 255, 0);
    pub const LIGHT_GREEN: Self = Self::from_rgb(0x90, 0xEE, 0x90);

    pub const DARK_BLUE: Self = Self::from_rgb(0, 0, 0x8B);
    pub const BLUE: Self = Self::from_rgb(0, 0, 255);
    pub const LIGHT_BLUE: Self = Self::from_rgb(0xAD, 0xD8, 0xE6);

    pub const fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        return Self { r, g, b };
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Decoration {
    pub pos: f64,
    pub rotate: f64,
    pub color: Color,
    pub size: f64,
    pub kind: DecorationKind,
}

#[derive(Debug, Clone, Copy)]
pub enum DecorationKind {
    Cross,
    Triangle,
    Box,
    Circle,
}
