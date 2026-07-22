pub fn digit_super(c: &char) -> char {
    match *c {
        '0' => '\u{2070}',
        '1' => '\u{00B9}',
        '2' => '\u{00B2}',
        '3' => '\u{00B3}',
        '4' => '\u{2074}',
        '5' => '\u{2075}',
        '6' => '\u{2076}',
        '7' => '\u{2077}',
        '8' => '\u{2078}',
        '9' => '\u{2079}',
        '-' => '\u{207B}',
        _ => unreachable!(),
    }
}

pub fn digit_sub(c: &char) -> char {
    match *c {
        '0' => '\u{2080}',
        '1' => '\u{2081}',
        '2' => '\u{2082}',
        '3' => '\u{2083}',
        '4' => '\u{2084}',
        '5' => '\u{2085}',
        '6' => '\u{2086}',
        '7' => '\u{2087}',
        '8' => '\u{2088}',
        '9' => '\u{2089}',
        '-' => '\u{208B}',
        _ => unreachable!(),
    }
}

pub fn index_super(i: i64) -> String {
    i.to_string().chars().map(|c| digit_super(&c)).collect::<String>()
}

pub fn index_sub(i: i64) -> String {
    i.to_string().chars().map(|c| digit_sub(&c)).collect::<String>()
}

pub fn indices_super(indices: &[i64]) -> String {
    indices
        .into_iter()
        .map(|i| index_super(*i))
        .collect::<Vec<_>>()
        .join("\u{0670}")
}

pub fn indices_sub(indices: &[i64]) -> String {
    indices
        .into_iter()
        .map(|i| index_sub(*i))
        .collect::<Vec<_>>()
        .join("\u{0656}")
}
