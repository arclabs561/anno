//! Language evidence for the dataset configurations whose publishers define each config as a language.
use super::dataset_registry::DatasetId;

// Names from https://datasets-server.huggingface.co/splits?dataset=unimelb-nlp%2Fwikiann .
// Legacy Wikipedia configurations without an unambiguous language-code mapping stay uncredited.
const WIKIANN_LANGUAGES: &str = "
ace af am an ang ar arc arz as ast ay az ba bar be bg bh bn bo br bs ca cdo ce ceb ckb co crh cs
csb cv cy da de diq dv el en eo es et eu ext fa fi fo fr frr fur fy ga gan gd gl gn gu hak he hi hr hsb
hu hy ia id ig ilo io is it ja jbo jv ka kk km kn ko ksh ku ky la lb li lij lmo ln lt lv mg mhr mi
min mk ml mn mr ms mt mwl my mzn nap nds ne nl nn no nov oc or os pa pdc pl pms pnb ps pt qu rm ro
ru rw sa sah scn sco sd sh si sk sl so sq sr su sv sw szl ta te tg th tk tl tr tt ug uk ur uz vec
vep vi vls vo wa war wuu xmf yi yo zea zh";
const WIKIANN_ALIASES: &[(&str, &str)] = &[
    ("bat-smg", "sgs"),
    ("be-x-old", "be-tarask"),
    ("cbk-zam", "cbk"),
    ("fiu-vro", "vro"),
    ("simple", "en"),
    ("zh-classical", "lzh"),
    ("zh-min-nan", "nan"),
    ("zh-yue", "yue"),
];

/// MasakhaNER 2.0 language codes.
///
/// These codes can be used to load specific language splits from MasakhaNER 2.0.
/// Example: `load_dataset("masakhane/masakhaner2", "yor")` for Yoruba.
pub const MASAKHANER2_LANGUAGES: &[(&str, &str)] = &[
    ("bam", "Bambara"),
    ("bbj", "Ghomala"),
    ("ewe", "Ewe"),
    ("fon", "Fon"),
    ("hau", "Hausa"),
    ("ibo", "Igbo"),
    ("kin", "Kinyarwanda"),
    ("lug", "Luganda"),
    ("luo", "Dholuo"),
    ("mos", "Mossi"),
    ("nya", "Chichewa"),
    ("pcm", "Nigerian Pidgin"),
    ("sna", "Shona"),
    ("swa", "Swahili"),
    ("tsn", "Setswana"),
    ("twi", "Twi"),
    ("wol", "Wolof"),
    ("xho", "Xhosa"),
    ("yor", "Yoruba"),
    ("zul", "Zulu"),
];

pub(crate) fn config_language(dataset: DatasetId, config: &str) -> Option<&'static str> {
    match dataset {
        DatasetId::WikiANN => WIKIANN_LANGUAGES
            .split_ascii_whitespace()
            .find(|&l| l == config)
            .or_else(|| {
                WIKIANN_ALIASES
                    .iter()
                    .find(|(name, _)| *name == config)
                    .map(|(_, lang)| *lang)
            }),
        DatasetId::MasakhaNER2 => MASAKHANER2_LANGUAGES
            .iter()
            .find(|(code, _)| *code == config)
            .map(|(code, _)| *code),
        _ => None,
    }
}

#[cfg(feature = "eval")]
pub(crate) fn verified_config_languages() -> impl Iterator<Item = &'static str> {
    WIKIANN_LANGUAGES
        .split_ascii_whitespace()
        .chain(WIKIANN_ALIASES.iter().map(|(_, lang)| *lang))
        .chain(MASAKHANER2_LANGUAGES.iter().map(|(code, _)| *code))
}

/// Compare provider tags using the ISO 639-1 equivalents of MasakhaNER2's ISO 639-3 tags.
/// The loader keeps the publisher config in the resolved source URL.
pub(crate) fn comparison_tag(tag: &str) -> &str {
    match tag {
        "bam" => "bm",
        "ewe" => "ee",
        "hau" => "ha",
        "ibo" => "ig",
        "kin" => "rw",
        "lug" => "lg",
        "nya" => "ny",
        "sna" => "sn",
        "swa" => "sw",
        "tsn" => "tn",
        "twi" => "tw",
        "wol" => "wo",
        "xho" => "xh",
        "yor" => "yo",
        "zul" => "zu",
        _ => tag,
    }
}

/// Resolve a sampling-language alias to this provider's config for cold downloads.
pub(crate) fn provider_config(dataset: DatasetId, pin: &str) -> &str {
    match dataset {
        DatasetId::MasakhaNER2 => MASAKHANER2_LANGUAGES
            .iter()
            .find(|(key, _)| comparison_tag(key) == comparison_tag(pin))
            .map_or(pin, |(key, _)| *key),
        DatasetId::WikiANN => {
            let tag = comparison_tag(pin);
            if WIKIANN_LANGUAGES
                .split_ascii_whitespace()
                .any(|config| config == tag)
            {
                tag
            } else {
                WIKIANN_ALIASES
                    .iter()
                    .find(|(_, lang)| *lang == tag)
                    .map_or(pin, |(key, _)| *key)
            }
        }
        _ => pin,
    }
}

/// Which observed inputs may pay down the Estimate scheduler's coverage budget.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) enum CoverageLanguage {
    #[default]
    Any,
    Unknown,
    Verified(std::collections::BTreeSet<String>),
}

impl CoverageLanguage {
    pub(crate) fn for_panel(
        dataset: DatasetId,
        pins: Option<&std::collections::BTreeSet<String>>,
        observed: Option<&str>,
    ) -> Self {
        let variable = dataset.language() == "mul" || dataset == DatasetId::MultiNERD;
        if pins.is_none() && !variable {
            return Self::Any;
        }
        if let Some(observed) = observed.filter(|lang| {
            pins.is_none_or(|pins| {
                pins.iter()
                    .any(|pin| comparison_tag(pin) == comparison_tag(lang))
            })
        }) {
            return Self::Verified([comparison_tag(observed).to_owned()].into());
        }
        if let Some(pins) = pins {
            return Self::Verified(
                pins.iter()
                    .map(|pin| comparison_tag(pin).to_owned())
                    .collect(),
            );
        }
        Self::Unknown
    }

    /// Once the input is loaded, an unverified result belongs only to the unknown slice.
    /// A prospective pin is useful before preflight but cannot establish what a backend saw.
    pub(crate) fn for_loaded(
        dataset: DatasetId,
        pins: Option<&std::collections::BTreeSet<String>>,
        observed: Option<&str>,
    ) -> Self {
        if observed.is_none() && pins.is_some() {
            Self::Unknown
        } else {
            Self::for_panel(dataset, pins, observed)
        }
    }

    pub(crate) fn includes(&self, observed: Option<&str>) -> bool {
        match self {
            Self::Any => true,
            Self::Unknown => observed.is_none(),
            Self::Verified(langs) => {
                observed.is_some_and(|lang| langs.contains(comparison_tag(lang)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observed_language_aliases_are_scoped_to_the_source_config() {
        assert_eq!(provider_config(DatasetId::MasakhaNER2, "yo"), "yor");
        assert_eq!(provider_config(DatasetId::WikiANN, "yor"), "yo");
        assert_eq!(provider_config(DatasetId::WikiANN, "yue"), "zh-yue");
        assert_eq!(provider_config(DatasetId::WikiANN, "en"), "en");
        let pins = ["yo".to_owned()].into();
        let scope = CoverageLanguage::for_panel(DatasetId::MasakhaNER2, Some(&pins), Some("yor"));
        assert!(scope.includes(Some("yo")) && scope.includes(Some("yor")));
        assert!(!scope.includes(Some("bam")) && !scope.includes(None));
        assert_eq!(
            CoverageLanguage::for_panel(DatasetId::WikiGold, None, Some("en")),
            CoverageLanguage::Any
        );
        assert_eq!(
            CoverageLanguage::for_panel(DatasetId::WikiANN, None, None),
            CoverageLanguage::Unknown
        );
    }
}
