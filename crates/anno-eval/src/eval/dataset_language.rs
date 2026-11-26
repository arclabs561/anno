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
