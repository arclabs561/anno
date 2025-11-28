//! Specialized domain synthetic datasets.
//!
//! Contains datasets for: sports, politics, ecommerce, travel, weather,
//! academic, food, real_estate, cybersecurity, multilingual, and globally diverse.

use super::super::types::helpers::entity;
use super::super::types::{AnnotatedExample, Difficulty, Domain};
use crate::EntityType;

/// Sports domain dataset.
pub fn sports_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "LeBron James scored 35 points as the Lakers defeated the Celtics 112-108.".into(),
            entities: vec![
                entity("LeBron James", EntityType::Person, 0),
                entity("Lakers", EntityType::Organization, 37),
                entity("Celtics", EntityType::Organization, 57),
            ],
            domain: Domain::Sports,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Manchester United signed Cristiano Ronaldo for $15 million.".into(),
            entities: vec![
                entity("Manchester United", EntityType::Organization, 0),
                entity("Cristiano Ronaldo", EntityType::Person, 25),
                entity("$15 million", EntityType::Money, 47),
            ],
            domain: Domain::Sports,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Serena Williams won Wimbledon on July 14, 2018 with a 67% first serve rate.".into(),
            entities: vec![
                entity("Serena Williams", EntityType::Person, 0),
                entity("Wimbledon", EntityType::Location, 20),
                entity("July 14, 2018", EntityType::Date, 33),
                entity("67%", EntityType::Percent, 54),
            ],
            domain: Domain::Sports,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Rafael Nadal defeated Roger Federer at Roland Garros in Paris.".into(),
            entities: vec![
                entity("Rafael Nadal", EntityType::Person, 0),
                entity("Roger Federer", EntityType::Person, 22),
                entity("Roland Garros", EntityType::Location, 39),
                entity("Paris", EntityType::Location, 56),
            ],
            domain: Domain::Sports,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Simone Biles won gold for the USA at the Tokyo Olympics.".into(),
            entities: vec![
                entity("Simone Biles", EntityType::Person, 0),
                entity("USA", EntityType::Organization, 30),
                entity("Tokyo", EntityType::Location, 41),
            ],
            domain: Domain::Sports,
            difficulty: Difficulty::Easy,
        },
    ]
}

/// Politics/government domain dataset.
pub fn politics_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "President Obama visited China to meet President Xi Jinping.".into(),
            entities: vec![
                entity("Obama", EntityType::Person, 10),
                entity("China", EntityType::Location, 24),
                entity("Xi Jinping", EntityType::Person, 48),
            ],
            domain: Domain::Politics,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "The United Nations held a summit in Geneva on March 15, 2024.".into(),
            entities: vec![
                entity("United Nations", EntityType::Organization, 4),
                entity("Geneva", EntityType::Location, 36),
                entity("March 15, 2024", EntityType::Date, 46),
            ],
            domain: Domain::Politics,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Senator Elizabeth Warren proposed a 2% wealth tax on billionaires.".into(),
            entities: vec![
                entity("Elizabeth Warren", EntityType::Person, 8),
                entity("2%", EntityType::Percent, 36),
            ],
            domain: Domain::Politics,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "NATO members met in Brussels to discuss Ukraine security.".into(),
            entities: vec![
                entity("NATO", EntityType::Organization, 0),
                entity("Brussels", EntityType::Location, 20),
                entity("Ukraine", EntityType::Location, 40),
            ],
            domain: Domain::Politics,
            difficulty: Difficulty::Medium,
        },
    ]
}

/// E-commerce domain dataset.
pub fn ecommerce_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "Amazon Prime Day sales reached $12.7 billion on July 12, 2023.".into(),
            entities: vec![
                entity("Amazon", EntityType::Organization, 0),
                entity("$12.7 billion", EntityType::Money, 31),
                entity("July 12, 2023", EntityType::Date, 48),
            ],
            domain: Domain::Ecommerce,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Shopify merchants generated $7B during Black Friday weekend.".into(),
            entities: vec![
                entity("Shopify", EntityType::Organization, 0),
                entity("$7B", EntityType::Money, 28),
            ],
            domain: Domain::Ecommerce,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Alibaba's Singles Day broke records with $84.5 billion in sales.".into(),
            entities: vec![
                entity("Alibaba", EntityType::Organization, 0),
                entity("$84.5 billion", EntityType::Money, 41),
            ],
            domain: Domain::Ecommerce,
            difficulty: Difficulty::Easy,
        },
    ]
}

/// Travel domain dataset.
pub fn travel_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "United Airlines flight UA100 departs from Los Angeles to Tokyo at 10:30 AM.".into(),
            entities: vec![
                entity("United Airlines", EntityType::Organization, 0),
                entity("Los Angeles", EntityType::Location, 42),
                entity("Tokyo", EntityType::Location, 57),
                entity("10:30 AM", EntityType::Date, 66),
            ],
            domain: Domain::Travel,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "The Hilton in Paris is $250 per night during Fashion Week.".into(),
            entities: vec![
                entity("Hilton", EntityType::Organization, 4),
                entity("Paris", EntityType::Location, 14),
                entity("$250", EntityType::Money, 23),
            ],
            domain: Domain::Travel,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Emirates offers direct flights from Dubai to New York in 14 hours.".into(),
            entities: vec![
                entity("Emirates", EntityType::Organization, 0),
                entity("Dubai", EntityType::Location, 36),
                entity("New York", EntityType::Location, 45),
            ],
            domain: Domain::Travel,
            difficulty: Difficulty::Easy,
        },
    ]
}

/// Weather domain dataset.
pub fn weather_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "Hurricane Maria made landfall in Puerto Rico on September 20, 2017.".into(),
            entities: vec![
                entity("Puerto Rico", EntityType::Location, 33),
                entity("September 20, 2017", EntityType::Date, 48),
            ],
            domain: Domain::Weather,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "The National Weather Service issued a flood warning for Miami.".into(),
            entities: vec![
                entity("National Weather Service", EntityType::Organization, 4),
                entity("Miami", EntityType::Location, 56),
            ],
            domain: Domain::Weather,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Temperatures in Death Valley reached 130°F on July 10, 2021.".into(),
            entities: vec![
                entity("Death Valley", EntityType::Location, 16),
                entity("July 10, 2021", EntityType::Date, 46),
            ],
            domain: Domain::Weather,
            difficulty: Difficulty::Medium,
        },
    ]
}

/// Academic domain dataset.
pub fn academic_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "Prof. Yoshua Bengio won the Turing Award alongside Geoffrey Hinton.".into(),
            entities: vec![
                entity("Prof. Yoshua Bengio", EntityType::Person, 0),
                entity("Geoffrey Hinton", EntityType::Person, 51),
            ],
            domain: Domain::Academic,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Stanford University received a $1 billion grant from the NIH.".into(),
            entities: vec![
                entity("Stanford University", EntityType::Organization, 0),
                entity("$1 billion", EntityType::Money, 31),
                entity("NIH", EntityType::Organization, 57),
            ],
            domain: Domain::Academic,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "The MIT CSAIL lab published breakthrough research in Nature.".into(),
            entities: vec![
                entity("MIT", EntityType::Organization, 4),
                entity("CSAIL", EntityType::Organization, 8),
                entity("Nature", EntityType::Organization, 53),
            ],
            domain: Domain::Academic,
            difficulty: Difficulty::Medium,
        },
    ]
}

/// Food domain dataset.
pub fn food_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "Chef Gordon Ramsay opened a new restaurant in Las Vegas.".into(),
            entities: vec![
                entity("Gordon Ramsay", EntityType::Person, 5),
                entity("Las Vegas", EntityType::Location, 46),
            ],
            domain: Domain::Food,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Chipotle Mexican Grill announced $2 billion in Q4 revenue.".into(),
            entities: vec![
                entity("Chipotle Mexican Grill", EntityType::Organization, 0),
                entity("$2 billion", EntityType::Money, 33),
            ],
            domain: Domain::Food,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Starbucks raised prices by 5% starting January 1, 2024.".into(),
            entities: vec![
                entity("Starbucks", EntityType::Organization, 0),
                entity("5%", EntityType::Percent, 27),
                entity("January 1, 2024", EntityType::Date, 39),
            ],
            domain: Domain::Food,
            difficulty: Difficulty::Easy,
        },
    ]
}

/// Real estate domain dataset.
pub fn real_estate_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "CBRE reported $500 million in commercial real estate sales in Manhattan.".into(),
            entities: vec![
                entity("CBRE", EntityType::Organization, 0),
                entity("$500 million", EntityType::Money, 14),
                entity("Manhattan", EntityType::Location, 62),
            ],
            domain: Domain::RealEstate,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Zillow listed the Beverly Hills mansion for $45 million.".into(),
            entities: vec![
                entity("Zillow", EntityType::Organization, 0),
                entity("Beverly Hills", EntityType::Location, 18),
                entity("$45 million", EntityType::Money, 44),
            ],
            domain: Domain::RealEstate,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Blackstone acquired the office tower in Chicago for $1.2 billion.".into(),
            entities: vec![
                entity("Blackstone", EntityType::Organization, 0),
                entity("Chicago", EntityType::Location, 40),
                entity("$1.2 billion", EntityType::Money, 52),
            ],
            domain: Domain::RealEstate,
            difficulty: Difficulty::Medium,
        },
    ]
}

/// Cybersecurity domain dataset.
pub fn cybersecurity_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "Microsoft patched CVE-2024-1234 affecting Windows 11 on Patch Tuesday.".into(),
            entities: vec![
                entity("Microsoft", EntityType::Organization, 0),
                entity("Windows 11", EntityType::Organization, 42),
            ],
            domain: Domain::Cybersecurity,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "CrowdStrike detected APT29 activity targeting NATO infrastructure.".into(),
            entities: vec![
                entity("CrowdStrike", EntityType::Organization, 0),
                entity("NATO", EntityType::Organization, 46),
            ],
            domain: Domain::Cybersecurity,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "The FBI and CISA issued a joint advisory about ransomware targeting healthcare.".into(),
            entities: vec![
                entity("FBI", EntityType::Organization, 4),
                entity("CISA", EntityType::Organization, 12),
            ],
            domain: Domain::Cybersecurity,
            difficulty: Difficulty::Medium,
        },
    ]
}

/// Multilingual dataset with native scripts.
pub fn multilingual_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "田中太郎さんは東京で働いています。".into(),
            entities: vec![
                entity("田中太郎", EntityType::Person, 0),
                entity("東京", EntityType::Location, 7),
            ],
            domain: Domain::Multilingual,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "الرئيس الأمريكي زار القاهرة في يناير.".into(),
            entities: vec![entity("القاهرة", EntityType::Location, 20)],
            domain: Domain::Multilingual,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "Präsident Steinmeier besuchte Berlin am 15. März.".into(),
            entities: vec![
                entity("Steinmeier", EntityType::Person, 10),
                entity("Berlin", EntityType::Location, 30),
                entity("15. März", EntityType::Date, 40),
            ],
            domain: Domain::Multilingual,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "北京大学的李教授获得了诺贝尔奖。".into(),
            entities: vec![
                entity("北京大学", EntityType::Organization, 0),
                entity("李", EntityType::Person, 5),
            ],
            domain: Domain::Multilingual,
            difficulty: Difficulty::Hard,
        },
    ]
}

/// Globally diverse dataset for demographic bias testing.
pub fn globally_diverse_dataset() -> Vec<AnnotatedExample> {
    vec![
        // African names
        AnnotatedExample {
            text: "Chidi Okonkwo is the CEO of Lagos Tech Solutions in Nigeria.".into(),
            entities: vec![
                entity("Chidi Okonkwo", EntityType::Person, 0),
                entity("Lagos Tech Solutions", EntityType::Organization, 28),
                entity("Nigeria", EntityType::Location, 52),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Medium,
        },
        // South Asian names
        AnnotatedExample {
            text: "Dr. Priya Sharma presented research at IIT Delhi on February 15, 2024.".into(),
            entities: vec![
                entity("Dr. Priya Sharma", EntityType::Person, 0),
                entity("IIT Delhi", EntityType::Organization, 39),
                entity("February 15, 2024", EntityType::Date, 52),
            ],
            domain: Domain::Academic,
            difficulty: Difficulty::Medium,
        },
        // East Asian names
        AnnotatedExample {
            text: "Wei Wang and Li Zhang lead Tsinghua University's AI research team.".into(),
            entities: vec![
                entity("Wei Wang", EntityType::Person, 0),
                entity("Li Zhang", EntityType::Person, 13),
                entity("Tsinghua University", EntityType::Organization, 27),
            ],
            domain: Domain::Academic,
            difficulty: Difficulty::Medium,
        },
        // Middle Eastern names
        AnnotatedExample {
            text: "Ahmed Hassan founded Dubai Innovations with backing from Abu Dhabi.".into(),
            entities: vec![
                entity("Ahmed Hassan", EntityType::Person, 0),
                entity("Dubai Innovations", EntityType::Organization, 21),
                entity("Abu Dhabi", EntityType::Location, 57),
            ],
            domain: Domain::Financial,
            difficulty: Difficulty::Medium,
        },
        // Hispanic/Latino names
        AnnotatedExample {
            text: "José García and María Rodriguez lead UNAM's research in Mexico City.".into(),
            entities: vec![
                entity("José García", EntityType::Person, 0),
                entity("María Rodriguez", EntityType::Person, 16),
                entity("UNAM", EntityType::Organization, 37),
                entity("Mexico City", EntityType::Location, 56),
            ],
            domain: Domain::Academic,
            difficulty: Difficulty::Medium,
        },
        // Eastern European names
        AnnotatedExample {
            text: "Ivan Petrov met Olga Ivanova in Moscow at the Kremlin.".into(),
            entities: vec![
                entity("Ivan Petrov", EntityType::Person, 0),
                entity("Olga Ivanova", EntityType::Person, 16),
                entity("Moscow", EntityType::Location, 32),
                entity("Kremlin", EntityType::Location, 46),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Medium,
        },
        // Mixed/intersectional
        AnnotatedExample {
            text: "Priya Sharma from Mumbai met Wei Wang from Beijing at MIT.".into(),
            entities: vec![
                entity("Priya Sharma", EntityType::Person, 0),
                entity("Mumbai", EntityType::Location, 18),
                entity("Wei Wang", EntityType::Person, 29),
                entity("Beijing", EntityType::Location, 43),
                entity("MIT", EntityType::Organization, 54),
            ],
            domain: Domain::Academic,
            difficulty: Difficulty::Medium,
        },
    ]
}

/// Hard examples for underrepresented domains.
pub fn hard_domain_examples() -> Vec<AnnotatedExample> {
    vec![
        // Technical (Hard)
        AnnotatedExample {
            text:
                "The CVE-2024-1234 vulnerability in OpenSSL 3.0.x affects nginx, Apache, and HAProxy."
                    .into(),
            entities: vec![
                entity("OpenSSL", EntityType::Organization, 35),
                entity("nginx", EntityType::Organization, 57),
                entity("Apache", EntityType::Organization, 64),
                entity("HAProxy", EntityType::Organization, 76),
            ],
            domain: Domain::Technical,
            difficulty: Difficulty::Hard,
        },
        // Travel (Hard)
        AnnotatedExample {
            text:
                "Connecting via FRA (Frankfurt) to SIN (Singapore) then SYD (Sydney) on LH/SQ codeshare."
                    .into(),
            entities: vec![
                entity("FRA", EntityType::Location, 15),
                entity("Frankfurt", EntityType::Location, 20),
                entity("SIN", EntityType::Location, 34),
                entity("Singapore", EntityType::Location, 39),
                entity("SYD", EntityType::Location, 55),
                entity("Sydney", EntityType::Location, 60),
            ],
            domain: Domain::Travel,
            difficulty: Difficulty::Hard,
        },
        // Entertainment (Hard)
        AnnotatedExample {
            text:
                "Director Christopher Nolan's Oppenheimer starring Cillian Murphy won at both the Oscars and BAFTAs."
                    .into(),
            entities: vec![
                entity("Christopher Nolan", EntityType::Person, 9),
                entity("Cillian Murphy", EntityType::Person, 50),
                entity("Oscars", EntityType::Organization, 81),
                entity("BAFTAs", EntityType::Organization, 92),
            ],
            domain: Domain::Entertainment,
            difficulty: Difficulty::Hard,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sports_dataset_not_empty() {
        assert!(!sports_dataset().is_empty());
    }

    #[test]
    fn test_politics_dataset_not_empty() {
        assert!(!politics_dataset().is_empty());
    }

    #[test]
    fn test_multilingual_dataset_not_empty() {
        assert!(!multilingual_dataset().is_empty());
    }

    #[test]
    fn test_globally_diverse_not_empty() {
        assert!(!globally_diverse_dataset().is_empty());
    }
}

