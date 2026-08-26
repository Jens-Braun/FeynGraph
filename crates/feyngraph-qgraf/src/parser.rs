use std::ops::Range;

use util::{
    HashMap,
    parsing::{
        DiagnosticSink, InvalidInt, PSLexer, Sep, Spanned,
        diagnostic::{DiagnosticContainer, InvalidString, UnexpectedInput},
    },
};

use crate::{
    ConstValue, QGRAFModel,
    diagnostic::{
        IntConversion, InvalidIdent, InvalidThird, UnclosedList, UnclosedStatement, UndefinedParticle, UnknownStatement,
    },
    lexer::{Lexer, Token},
};

pub(crate) struct QGRAFParser<'s> {
    lex: PSLexer<'s, Token<'s>>,
    sink: DiagnosticSink,
    model: QGRAFModel,

    failed_fields: Vec<&'s str>,
    conversion_warning: bool,
}

impl<'s> QGRAFParser<'s> {
    pub(crate) fn parse(name: &str, src: &'s str) -> DiagnosticContainer<QGRAFModel> {
        let mut parser = Self {
            lex: Lexer::new(src),
            sink: DiagnosticSink::new(),
            model: QGRAFModel::new(),

            failed_fields: Vec::new(),
            conversion_warning: true,
        };

        DiagnosticContainer::new(parser.model().map(|()| parser.model), parser.sink, name, src)
    }

    fn model(&mut self) -> Option<()> {
        let mut success = true;
        while self.lex.peek().is_some() {
            match self.statement() {
                Some(_) => (),
                None => success = false,
            }
        }
        if success {
            return Some(());
        } else {
            return None;
        }
    }

    fn statement(&mut self) -> Option<()> {
        self.sink.set_checkpoint();
        let open = match &self.lex.peek_report("statement", &mut self.sink)?.inner {
            Ok(Token::BrackOpen) => {
                let tok = self.lex.next().unwrap();
                tok.span
            }
            _ => {
                let span = self.consume_until_statement();
                self.sink.push(UnexpectedInput::expected("statement", span));
                return None;
            }
        };
        let res = self.statement_inner();
        let close = match &self.lex.peek_report("']'", &mut self.sink)?.inner {
            Ok(Token::BrackClose) => {
                let tok = self.lex.next().unwrap();
                tok.span
            }
            _ => {
                let span = self.consume_until_statement();
                self.sink.push(UnclosedStatement::new(open, span));
                return None;
            }
        };
        match res {
            None => {
                self.sink.restore_checkpoint();
                self.sink.push(UnknownStatement::new(open.start..close.end));
                return Some(());
            }
            Some(false) => return None,
            Some(true) => return Some(()),
        }
    }

    fn statement_inner(&mut self) -> Option<bool> {
        let mut buf = Vec::new();
        self.ident_list(&mut buf, Some(2))?;
        self.check_statement_end()?;
        match self.lex.peek_report_list(&["int", "ident"], &mut self.sink)?.inner {
            Ok(Token::Int(_) | Token::Plus | Token::Minus) => match self.continue_field(&buf) {
                None => {
                    self.consume_statement();
                    self.failed_fields.push(buf[0].inner);
                    self.failed_fields.push(buf[1].inner);
                    return Some(false);
                }
                Some(_) => return Some(true),
            },
            Ok(Token::Ident(_)) => match self.continue_vertex(buf) {
                None => {
                    self.consume_statement();
                    return Some(false);
                }
                Some(_) => return Some(true),
            },
            _ => {
                let span = self.consume_until_sep();
                let _ = self.consume_statement();
                self.sink.push(InvalidThird::new(span));
                return Some(false);
            }
        }
    }

    fn continue_field(&mut self, buf: &Vec<Spanned<&'s str>>) -> Option<()> {
        let fermi = match self.lex.peek_report("field sign", &mut self.sink).unwrap().tup() {
            (span, Ok(Token::Int(_))) => {
                let i = self.int()?;
                match i {
                    1 => false,
                    -1 => true,
                    _ => {
                        self.sink.push(InvalidThird::new(span));
                        return None;
                    }
                }
            }
            (_, Ok(Token::Plus)) => {
                let _ = self.lex.next();
                false
            }
            (_, Ok(Token::Minus)) => {
                let _ = self.lex.next();
                true
            }
            _ => unreachable!(),
        };
        let mut keywords = Vec::new();
        let mut kw_success = true;
        if let Ok(tok) = &self.lex.peek_report_list(&["','", "';'", "']'"], &mut self.sink)?.inner
            && matches!(tok, Token::Comma)
        {
            let _ = self.lex.next();
            if self.ident_list(&mut keywords, None).is_none() {
                kw_success = false;
            }
        }
        let keywords = keywords.into_iter().map(|s| s.to_owned()).collect();
        match self.lex.peek_report_list(&["';'", "']'"], &mut self.sink)?.tup() {
            (_, Ok(Token::Semicolon)) => {
                let _ = self.lex.next();
            }
            (_, Ok(Token::BrackClose)) => {
                self.model
                    .push_particle(buf[0].inner, buf[1].inner, fermi, keywords, HashMap::default());
                return Some(());
            }
            (span, _) => {
                self.sink.push(UnexpectedInput::expected("';'", span.clone()));
                return None;
            }
        }
        let mut props = Vec::new();
        self.prop_list(&mut props)?;
        if kw_success {
            self.model.push_particle(
                buf[0].inner,
                buf[1].inner,
                fermi,
                keywords,
                props.into_iter().map(|(s, v)| (s.to_owned(), v)).collect(),
            );
            return Some(());
        } else {
            return None;
        }
    }

    fn continue_vertex(&mut self, mut buf: Vec<Spanned<&'s str>>) -> Option<()> {
        self.ident_list(&mut buf, None)?;
        let mut success = true;
        for p in buf.iter() {
            if !self.model.particles.contains_key(p.inner) {
                if !self.failed_fields.contains(&p.inner) {
                    self.sink.push(UndefinedParticle::new(p.inner, p.span.clone()));
                }
                success = false;
            }
        }
        match self.lex.peek_report("';'", &mut self.sink)?.tup() {
            (_, Ok(Token::Semicolon)) => {
                let _ = self.lex.next();
            }
            (_, Ok(Token::BrackClose)) => {
                if success {
                    self.model
                        .push_vertex(
                            format!("V_{}", self.model.n_vertices()),
                            buf.into_iter().map(|s| s.to_owned()).collect(),
                            HashMap::default(),
                            HashMap::default(),
                        )
                        .unwrap();
                    return Some(());
                } else {
                    return None;
                }
            }
            (span, _) => {
                self.sink.push(UnexpectedInput::expected("';'", span.clone()));
                return None;
            }
        }
        let mut props = Vec::new();
        self.prop_list(&mut props)?;
        let couplings = props
            .iter()
            .filter_map(|(p, v)| match v {
                ConstValue::Int(i) if *i > 0 => Some(((*p).to_owned(), *i as usize)),
                _ => None,
            })
            .collect();
        let constants = props
            .into_iter()
            .filter_map(|(p, v)| match v {
                ConstValue::Int(i) if i > 0 => None,
                _ => Some((p.to_owned(), v)),
            })
            .collect();
        if success {
            self.model
                .push_vertex(
                    format!("V_{}", self.model.n_vertices()),
                    buf.into_iter().map(|s| s.to_owned()).collect(),
                    couplings,
                    constants,
                )
                .unwrap();
            return Some(());
        } else {
            return None;
        }
    }

    fn ident_list(&mut self, buf: &mut Vec<Spanned<&'s str>>, n: Option<usize>) -> Option<()> {
        let m = n.unwrap_or(usize::MAX);
        let mut i = 0;
        let mut success = true;
        while i < m
            && let Some(_) = self.lex.peek()
        {
            self.check_statement_end()?;
            let id = self.ident();
            if let Some(id) = id {
                buf.push(id);
            } else {
                success = false;
            }
            self.check_statement_end()?;
            match &self.lex.peek_report("','", &mut self.sink)?.inner {
                Ok(Token::Comma) => {
                    let _ = self.lex.next();
                }
                Ok(tok) if tok.is_sep() => break,
                _ => {
                    success = false;
                    let span = self.consume_until_sep();
                    self.sink.push(UnexpectedInput::expected("','", span));
                }
            }
            i += 1;
        }
        if !success {
            return None;
        } else {
            if let Some(n) = n
                && i != n
            {
                return None;
            }
            return Some(());
        }
    }

    fn prop_list(&mut self, buf: &mut Vec<(&'s str, ConstValue)>) -> Option<()> {
        let mut success = true;
        while self.lex.peek().is_some() {
            if let Some(val) = self.property() {
                buf.push(val);
            } else {
                success = false;
            }
            match &self.lex.peek_report("','", &mut self.sink)?.inner {
                Ok(Token::Comma) => {
                    let _ = self.lex.next();
                }
                Ok(tok) if tok.is_sep() => break,
                _ => {
                    success = false;
                    let span = self.consume_until_sep();
                    self.sink.push(UnexpectedInput::expected("','", span));
                }
            }
        }
        if !success {
            return None;
        } else {
            return Some(());
        }
    }

    fn property(&mut self) -> Option<(&'s str, ConstValue)> {
        if let Some(ident) = self.ident() {
            match self.lex.peek_report("=", &mut self.sink)?.inner {
                Ok(Token::Eq) => {
                    let _ = self.lex.next();
                    if let Some(val) = self.constant() {
                        return Some((ident.inner, val));
                    }
                }
                _ => {
                    let span = self.consume_until_sep();
                    self.sink.push(UnexpectedInput::expected("'='", span));
                }
            }
        }
        return None;
    }

    fn constant(&mut self) -> Option<ConstValue> {
        match &self
            .lex
            .peek_report_list(&["integer", "string", "list"], &mut self.sink)?
            .inner
        {
            Ok(Token::Int(_)) | Ok(Token::String(_)) => return self.literal(),
            Ok(Token::ParenOpen) => return self.list(),
            _ => {
                let span = self.consume_until_sep();
                self.sink
                    .push(UnexpectedInput::expected_list(&["integer", "string", "list"], span));
                return None;
            }
        }
    }

    fn list(&mut self) -> Option<ConstValue> {
        match &self.lex.peek_report("list", &mut self.sink)?.inner {
            Ok(Token::ParenOpen) => {
                let start = self.lex.next().unwrap().span;
                let mut res = Vec::new();
                let mut success = true;
                while self.lex.peek().is_some() {
                    match self.literal() {
                        Some(v) => res.push(v),
                        None => success = false,
                    }
                    match self.lex.peek_report("','", &mut self.sink)?.tup() {
                        (_, Ok(Token::Comma)) => {
                            let _ = self.lex.next();
                        }
                        (span, Ok(Token::BrackClose)) => {
                            self.sink.push(UnclosedList::new(start, span));
                            break;
                        }
                        (_, Ok(Token::ParenClose)) => {
                            let _ = self.lex.next();
                            break;
                        }
                        (_, Ok(tok)) if tok.is_sep() => break,
                        _ => {
                            success = false;
                            let span = self.consume_until_sep();
                            self.sink.push(UnexpectedInput::expected("','", span));
                        }
                    }
                }
                if success {
                    return Some(ConstValue::List(res));
                } else {
                    return None;
                }
            }
            _ => {
                let span = self.consume_until_sep();
                self.sink.push(UnexpectedInput::expected("list", span));
                return None;
            }
        }
    }

    fn literal(&mut self) -> Option<ConstValue> {
        match &self.lex.peek_report_list(&["integer", "string"], &mut self.sink)? {
            Spanned { inner: Err(_), .. } => {
                let span = self.consume_until_sep();
                self.sink
                    .push(UnexpectedInput::expected_list(&["integer", "string"], span));
                return None;
            }
            Spanned { span, inner: Ok(t) } => match t {
                Token::Int(_) => return self.int().map(|i| ConstValue::Int(i)),
                Token::String(_) => {
                    let span = span.clone();
                    let s = self.string()?;
                    if let Ok(i) = s.parse::<i64>() {
                        if self.conversion_warning {
                            self.sink.push(IntConversion::new(span));
                            self.conversion_warning = false;
                        }
                        return Some(ConstValue::Int(i));
                    }
                    return Some(ConstValue::String(s.to_owned()));
                }
                _ => {
                    let span = self.consume_until_sep();
                    self.sink
                        .push(UnexpectedInput::expected_list(&["integer", "string"], span));
                    return None;
                }
            },
        }
    }

    fn int(&mut self) -> Option<i64> {
        match &self.lex.peek_report("integer", &mut self.sink)?.inner {
            Err(_) => {
                let span = self.consume_until_sep();
                self.sink.push(InvalidInt::new(span));
                return None;
            }
            Ok(t) => match t {
                Token::Int(_) => {
                    let i = self.lex.next().unwrap().inner.unwrap().int();
                    if self.lex.peek_sep() || self.lex.peek().is_none() {
                        return Some(i);
                    } else {
                        return None;
                    }
                }
                _ => {
                    let span = self.consume_until_sep();
                    self.sink.push(UnexpectedInput::expected("integer", span));
                    return None;
                }
            },
        }
    }

    fn string(&mut self) -> Option<&'s str> {
        match &self.lex.peek_report("string", &mut self.sink)?.inner {
            Err(_) => {
                let span = self.consume_until_sep();
                self.sink.push(InvalidString::new(span));
                return None;
            }
            Ok(t) => match t {
                Token::String(_) => {
                    let s = self.lex.next().unwrap().inner.unwrap().str();
                    if self.lex.peek_sep() || self.lex.peek().is_none() {
                        return Some(s);
                    } else {
                        return None;
                    }
                }
                _ => {
                    let span = self.consume_until_sep();
                    self.sink.push(UnexpectedInput::expected("string", span));
                    return None;
                }
            },
        }
    }

    fn ident(&mut self) -> Option<Spanned<&'s str>> {
        match &self.lex.peek_report("ident", &mut self.sink)?.inner {
            Err(_) => {
                let span = self.consume_until_sep();
                self.sink.push(InvalidIdent::new(span));
                return None;
            }
            Ok(t) => match t {
                Token::Ident(_) => {
                    let Spanned { span, inner } = self.lex.next().unwrap();
                    let s = inner.unwrap().ident();
                    if self.lex.peek_sep()
                        || matches!(
                            self.lex.peek(),
                            None | Some(Spanned {
                                inner: Ok(Token::Eq),
                                ..
                            })
                        )
                    {
                        return Some(Spanned { inner: s, span });
                    } else {
                        return None;
                    }
                }
                _ => {
                    let span = self.consume_until_sep();
                    self.sink.push(UnexpectedInput::expected("ident", span));
                    return None;
                }
            },
        }
    }

    fn consume_until_sep(&mut self) -> Range<usize> {
        let mut span = self.lex.span();
        match self.lex.peek() {
            None => return span,
            Some(t) => span = t.span.clone(),
        }
        while self.lex.peek().is_some() && !self.lex.peek_sep() {
            let t = self.lex.next().unwrap();
            span.end = t.span.end;
        }
        return span;
    }

    fn consume_statement(&mut self) -> Range<usize> {
        let mut span = self.lex.span();
        match self.lex.peek() {
            None => return span,
            Some(t) => span = t.span.clone(),
        }
        while let Some(tok) = self.lex.peek()
            && !matches!(tok.inner, Ok(Token::BrackOpen | Token::BrackClose))
        {
            let t = self.lex.next().unwrap();
            span.end = t.span.end;
        }
        return span;
    }

    fn consume_until_statement(&mut self) -> Range<usize> {
        let mut span = self.lex.span();
        match self.lex.peek() {
            None => return span,
            Some(t) => span = t.span.clone(),
        }
        while let Some(tok) = self.lex.peek()
            && !matches!(tok.inner, Ok(Token::BrackOpen))
        {
            let t = self.lex.next().unwrap();
            span.end = t.span.end;
        }
        return span;
    }

    fn check_statement_end(&mut self) -> Option<()> {
        match self.lex.peek() {
            None
            | Some(Spanned {
                inner: Ok(Token::BrackClose),
                ..
            }) => return None,
            _ => return Some(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn qgraf_parse_qcd_test() {
        let source =
            std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/models/qcd.qgraf")).unwrap();
        let m = QGRAFParser::parse("qcd.qgraf", &source);
        #[cfg(feature = "emit_test_reports")]
        m.emit().unwrap();
        assert!(m.is_some());
    }

    #[test]
    fn qgraf_parse_sm_test() {
        let source =
            std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/models/sm.qgraf")).unwrap();
        let m = QGRAFParser::parse("sm.qgraf", &source);
        #[cfg(feature = "emit_test_reports")]
        m.emit().unwrap();
        assert!(m.is_some());
    }
}
