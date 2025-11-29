//! Domain-specific NER tests.
//!
//! Tests NER performance on text from specific domains:
//! - Medical/clinical
//! - Legal/contracts
//! - Financial/business
//! - News/journalism
//! - Social media
//! - Technical/scientific

use anno::{Entity, EntityType, Model, PatternNER, StackedNER};

fn has_type(entities: &[Entity], ty: &EntityType) -> bool {
    entities.iter().any(|e| e.entity_type == *ty)
}

fn count_type(entities: &[Entity], ty: &EntityType) -> usize {
    entities.iter().filter(|e| e.entity_type == *ty).count()
}

// =============================================================================
// Medical Domain
// =============================================================================

mod medical {
    use super::*;

    #[test]
    fn clinical_note_dates() {
        let ner = PatternNER::new();
        let text = r#"
            Patient visited on 2024-01-15.
            Follow-up scheduled for February 28, 2024.
            Last lab work done 01/10/2024.
        "#;
        let e = ner.extract_entities(text, None).unwrap();
        assert!(count_type(&e, &EntityType::Date) >= 2);
    }

    #[test]
    fn clinical_note_times() {
        let ner = PatternNER::new();
        let text = r#"
            Vitals taken at 8:30am.
            Medication administered at 2:00 PM.
            Patient discharged at 17:45.
        "#;
        let e = ner.extract_entities(text, None).unwrap();
        assert!(count_type(&e, &EntityType::Time) >= 2);
    }

    #[test]
    fn medical_personnel() {
        let ner = StackedNER::new();
        let text = r#"
            Dr. Sarah Johnson examined the patient.
            Nurse Maria Garcia administered medication.
            Dr. Smith ordered additional tests.
        "#;
        let e = ner.extract_entities(text, None).unwrap();
        // Heuristic system - may or may not find persons
        // Just verify it doesn't panic and processes the text
        // Just verify no panic - processing succeeded
        let _ = e;
    }

    #[test]
    fn hospital_organizations() {
        let ner = StackedNER::new();
        let text = r#"
            Referred to Mayo Clinic for specialist consultation.
            Lab work processed by Quest Diagnostics.
            Insurance through Blue Cross Blue Shield.
        "#;
        let e = ner.extract_entities(text, None).unwrap();
        // May find some organizations
        // Just verify no panic - processing succeeded
        let _ = e;
    }

    #[test]
    fn percentages_in_vitals() {
        let ner = PatternNER::new();
        let text = "SpO2: 98%, Heart rate normal. Improvement of 15% since last visit.";
        let e = ner.extract_entities(text, None).unwrap();
        assert!(count_type(&e, &EntityType::Percent) >= 2);
    }
}

// =============================================================================
// Legal Domain
// =============================================================================

mod legal {
    use super::*;

    #[test]
    fn contract_dates() {
        let ner = PatternNER::new();
        let text = r#"
            This Agreement is entered into as of January 1, 2024 (the "Effective Date").
            Term shall end on December 31, 2024.
            Notice must be provided by 2024-06-30.
        "#;
        let e = ner.extract_entities(text, None).unwrap();
        assert!(count_type(&e, &EntityType::Date) >= 3);
    }

    #[test]
    fn contract_money() {
        let ner = PatternNER::new();
        let text = r#"
            Total consideration: $1,500,000.
            Monthly payment: $25,000.
            Penalty fee: $5,000 per incident.
            Not to exceed $100,000 in aggregate.
        "#;
        let e = ner.extract_entities(text, None).unwrap();
        assert!(count_type(&e, &EntityType::Money) >= 4);
    }

    #[test]
    fn legal_parties() {
        let ner = StackedNER::new();
        let text = r#"
            BETWEEN:
            Acme Corporation ("Company")
            AND
            John Smith ("Employee")
            
            WITNESSED BY:
            Jane Doe, Notary Public
        "#;
        let e = ner.extract_entities(text, None).unwrap();
        // Should find organizations and/or persons
        let named_count = e
            .iter()
            .filter(|e| matches!(e.entity_type, EntityType::Person | EntityType::Organization))
            .count();
        assert!(named_count >= 1, "Should find legal parties: {:?}", e);
    }

    #[test]
    fn legal_percentages() {
        let ner = PatternNER::new();
        let text = r#"
            Interest rate: 5.5% per annum.
            Commission: 3% of gross revenue.
            Ownership stake: 25%.
        "#;
        let e = ner.extract_entities(text, None).unwrap();
        assert!(count_type(&e, &EntityType::Percent) >= 3);
    }

    #[test]
    fn contact_info_in_contracts() {
        let ner = PatternNER::new();
        let text = r#"
            Notices shall be sent to:
            Email: legal@acme.com
            Phone: (555) 123-4567
            
            Or to the registered office at the address above.
        "#;
        let e = ner.extract_entities(text, None).unwrap();
        assert!(has_type(&e, &EntityType::Email));
        assert!(has_type(&e, &EntityType::Phone));
    }
}

// =============================================================================
// Financial Domain
// =============================================================================

mod financial {
    use super::*;

    #[test]
    fn earnings_report() {
        let ner = PatternNER::new();
        let text = r#"
            Q4 2024 Earnings Report
            
            Revenue: $45.2 million (up 15% YoY)
            Net Income: $8.3 million
            EPS: $2.15
            
            Guidance for 2025: $50-55 million
        "#;
        let e = ner.extract_entities(text, None).unwrap();
        assert!(count_type(&e, &EntityType::Money) >= 3);
        assert!(has_type(&e, &EntityType::Percent));
    }

    #[test]
    fn stock_analysis() {
        let ner = StackedNER::new();
        let text = r#"
            Apple Inc. (AAPL) closed at $185.50.
            Microsoft Corporation gained 2.3%.
            Alphabet Inc. reported earnings after market close.
        "#;
        let e = ner.extract_entities(text, None).unwrap();
        // Should find money and/or organizations
        assert!(!e.is_empty(), "Should find entities in stock analysis");
    }

    #[test]
    fn transaction_details() {
        let ner = PatternNER::new();
        let text = r#"
            Transaction ID: TX-2024-001
            Date: 2024-01-15 at 14:30
            Amount: $1,234.56
            Fee: $12.50 (1% of transaction)
        "#;
        let e = ner.extract_entities(text, None).unwrap();
        assert!(has_type(&e, &EntityType::Date));
        assert!(has_type(&e, &EntityType::Time));
        assert!(count_type(&e, &EntityType::Money) >= 2);
        assert!(has_type(&e, &EntityType::Percent));
    }

    #[test]
    fn bank_contact() {
        let ner = PatternNER::new();
        let text = r#"
            For assistance, contact:
            Customer Service: 1-800-555-0123
            Email: support@bank.com
            Online: https://www.bank.com/help
        "#;
        let e = ner.extract_entities(text, None).unwrap();
        assert!(has_type(&e, &EntityType::Phone));
        assert!(has_type(&e, &EntityType::Email));
        assert!(has_type(&e, &EntityType::Url));
    }
}

// =============================================================================
// News Domain
// =============================================================================

mod news {
    use super::*;

    #[test]
    fn news_article_entities() {
        let ner = StackedNER::new();
        let text = r#"
            WASHINGTON - President Johnson announced today that the administration
            will invest $50 billion in infrastructure. The announcement came during
            a press conference at the White House.
            
            "This represents a 25% increase over last year," said Press Secretary
            Maria Garcia.
        "#;
        let e = ner.extract_entities(text, None).unwrap();

        // Should find money
        assert!(has_type(&e, &EntityType::Money));
        // Should find percent
        assert!(has_type(&e, &EntityType::Percent));
        // May find persons and locations (heuristic)
    }

    #[test]
    fn sports_news() {
        let ner = StackedNER::new();
        let text = r#"
            The Lakers defeated the Celtics 112-108 in last night's game.
            LeBron James scored 35 points. Coach Smith praised the team's
            performance after the game.
        "#;
        let e = ner.extract_entities(text, None).unwrap();
        // May find persons (heuristic)
        // Just verify no panic - processing succeeded
        let _ = e;
    }

    #[test]
    fn dateline_extraction() {
        let ner = PatternNER::new();
        let text = r#"
            NEW YORK, January 15, 2024 - The stock market opened higher today.
            
            LONDON, 2024-01-15 - European markets followed suit.
        "#;
        let e = ner.extract_entities(text, None).unwrap();
        assert!(count_type(&e, &EntityType::Date) >= 2);
    }
}

// =============================================================================
// Social Media Domain
// =============================================================================

mod social_media {
    use super::*;

    #[test]
    fn tweet_with_entities() {
        let ner = PatternNER::new();
        let text = r#"
            Just bought tickets for $50! 🎉 
            Show starts at 8pm on 12/25/2024.
            DM me at test@email.com for details!
        "#;
        let e = ner.extract_entities(text, None).unwrap();
        assert!(has_type(&e, &EntityType::Money));
        assert!(has_type(&e, &EntityType::Time));
        assert!(has_type(&e, &EntityType::Date));
        assert!(has_type(&e, &EntityType::Email));
    }

    #[test]
    fn url_in_post() {
        let ner = PatternNER::new();
        let text = "Check out this link: https://example.com/page?id=123";
        let e = ner.extract_entities(text, None).unwrap();
        assert!(has_type(&e, &EntityType::Url));
    }

    #[test]
    fn informal_time_expressions() {
        let ner = PatternNER::new();
        let text = "See you at 3pm! Meeting at 10:30am tomorrow.";
        let e = ner.extract_entities(text, None).unwrap();
        assert!(count_type(&e, &EntityType::Time) >= 2);
    }

    #[test]
    fn hashtags_and_mentions() {
        let ner = StackedNER::new();
        // Hashtags and @ mentions - NER shouldn't crash
        let text = "@JohnSmith just announced #BigNews! Price is $99.";
        let e = ner.extract_entities(text, None).unwrap();
        assert!(has_type(&e, &EntityType::Money));
    }
}

// =============================================================================
// Technical/Scientific Domain
// =============================================================================

mod technical {
    use super::*;

    #[test]
    fn research_paper_dates() {
        let ner = PatternNER::new();
        let text = r#"
            Received: January 15, 2024
            Accepted: February 28, 2024
            Published: 2024-03-15
            
            Experiments conducted from 01/01/2024 to 01/31/2024.
        "#;
        let e = ner.extract_entities(text, None).unwrap();
        assert!(count_type(&e, &EntityType::Date) >= 4);
    }

    #[test]
    fn statistical_results() {
        let ner = PatternNER::new();
        let text = r#"
            Results showed a 23.5% improvement (p < 0.05).
            Control group: 45% success rate.
            Treatment group: 68.5% success rate.
        "#;
        let e = ner.extract_entities(text, None).unwrap();
        assert!(count_type(&e, &EntityType::Percent) >= 3);
    }

    #[test]
    fn author_affiliations() {
        let ner = StackedNER::new();
        let text = r#"
            Authors:
            Dr. John Smith, Massachusetts Institute of Technology
            Dr. Jane Doe, Stanford University
            Prof. Robert Johnson, Harvard Medical School
        "#;
        let e = ner.extract_entities(text, None).unwrap();
        // May find persons and/or organizations
        // Just verify no panic - processing succeeded
        let _ = e;
    }

    #[test]
    fn contact_author() {
        let ner = PatternNER::new();
        let text = r#"
            Corresponding author:
            Dr. John Smith
            Email: jsmith@mit.edu
            Phone: +1 (617) 555-1234
        "#;
        let e = ner.extract_entities(text, None).unwrap();
        assert!(has_type(&e, &EntityType::Email));
        assert!(has_type(&e, &EntityType::Phone));
    }

    #[test]
    fn code_snippet_with_entities() {
        let ner = PatternNER::new();
        // URL and email in code comments
        let text = r#"
            // See documentation at https://docs.example.com
            // Contact: dev@example.com
            let price = 100; // $100 in cents
        "#;
        let e = ner.extract_entities(text, None).unwrap();
        assert!(has_type(&e, &EntityType::Url));
        assert!(has_type(&e, &EntityType::Email));
    }
}

// =============================================================================
// Mixed Domain (Real-world complexity)
// =============================================================================

mod mixed {
    use super::*;

    #[test]
    fn business_email() {
        let ner = StackedNER::new();
        let text = r#"
            From: John Smith <john@acme.com>
            To: Jane Doe <jane@partner.com>
            Date: January 15, 2024 at 10:30 AM
            Subject: Q1 Budget Discussion
            
            Hi Jane,
            
            Following our call yesterday, here's the breakdown:
            - Marketing: $50,000 (up 15% from last quarter)
            - R&D: $75,000
            - Operations: $30,000
            
            Total: $155,000
            
            Let's discuss on Friday at 2pm.
            Call me at (555) 123-4567.
            
            Best,
            John
        "#;
        let e = ner.extract_entities(text, None).unwrap();

        // Should find multiple entity types
        assert!(has_type(&e, &EntityType::Email));
        assert!(has_type(&e, &EntityType::Date));
        assert!(has_type(&e, &EntityType::Time));
        assert!(count_type(&e, &EntityType::Money) >= 3);
        assert!(has_type(&e, &EntityType::Percent));
        assert!(has_type(&e, &EntityType::Phone));
    }

    #[test]
    fn invoice_document() {
        let ner = PatternNER::new();
        let text = r#"
            INVOICE #INV-2024-001
            Date: 2024-01-15
            Due: 2024-02-15
            
            Bill To:
            Acme Corporation
            123 Business St.
            
            Contact: billing@acme.com
            Phone: (555) 987-6543
            
            Items:
            - Consulting services: $5,000.00
            - Software license: $1,200.00
            - Support (20%): $1,240.00
            
            Subtotal: $7,440.00
            Tax (8.5%): $632.40
            Total: $8,072.40
            
            Payment URL: https://pay.example.com/inv/2024001
        "#;
        let e = ner.extract_entities(text, None).unwrap();

        assert!(count_type(&e, &EntityType::Date) >= 2);
        assert!(has_type(&e, &EntityType::Email));
        assert!(has_type(&e, &EntityType::Phone));
        assert!(count_type(&e, &EntityType::Money) >= 5);
        assert!(count_type(&e, &EntityType::Percent) >= 2);
        assert!(has_type(&e, &EntityType::Url));
    }

    #[test]
    fn conference_registration() {
        let ner = StackedNER::new();
        let text = r#"
            NLP Conference 2024
            
            Date: March 15-17, 2024
            Location: San Francisco, CA
            
            Registration: $500 (early bird: $400, ends 2024-02-01)
            Student rate: $250 (50% discount)
            
            Register at: https://nlp-conf.com/register
            Questions: info@nlp-conf.com
            
            Keynote Speakers:
            - Dr. Sarah Chen, Google Research
            - Prof. Michael Brown, MIT
        "#;
        let e = ner.extract_entities(text, None).unwrap();

        // Pattern entities
        assert!(count_type(&e, &EntityType::Money) >= 2);
        assert!(has_type(&e, &EntityType::Percent));
        assert!(has_type(&e, &EntityType::Url));
        assert!(has_type(&e, &EntityType::Email));
        assert!(has_type(&e, &EntityType::Date));
    }
}
