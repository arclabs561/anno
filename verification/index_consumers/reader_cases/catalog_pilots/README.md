# Dated geography and music credit pilots

These readers answer two catalog questions from published records. The fixed panel is for regression and exploration. The cases were selected for their contrasts and supply no population-level accuracy estimate.

## Run offline

From this directory:

```sh
python -m unittest discover -s . -p 'test_*.py'
python reader.py area 2025-12 E04006085 2023-04-01
python reader.py billing release/ad6c808f-2948-478a-aad5-3e02a9692e67
python reader.py works recording/b040d89e-190f-47a0-af7f-20534101056c
python reader.py role work/dee610f0-e425-311f-be39-554747afbddf writer
```

### Which authority succeeds a cited district?

We projected the same five English origin codes from the ONS December 2023 and December 2025 Code History Database editions. For each origin, the collector reads **all** outgoing records in `Changes.csv` and fetches the associated origin and target rows from `ChangeHistory.csv`. Both editions gave the same five change rows, including a row with an empty target, and eight distinct areas. Differences in source date formats disappear after normalization. Each result identifies the edition and original code, even after a successor is found. [2023 product](https://www.data.gov.uk/dataset/994d4647-ad45-47c5-8e6e-5e2edc1a42e5/code-history-database-december-2023-for-the-uk) · [2025 product](https://www.data.gov.uk/dataset/4ec2169f-1a72-4c2d-be60-a74174902593/code-history-database-december-2025-for-the-uk)

| Origin code | Direct evidence at April 1, 2023 |
|---|---|
| Carlisle `E07000028`, Copeland `E07000029` | Each terminated on March 31. Each has a dated change from a district (E07) to Cumberland `E06000063`, a unitary authority (E06). |
| Attleborough `E04006085` | Two new parish codes, Attleborough and Great Ellingham. The name alone would choose only one. |
| Cumbria `E10000006` | The county terminated on March 31. Its change row naming Order 331/2022 leaves the successor and effective date blank. |
| Norfolk `E10000020` | The code remained in operation in both editions. |

An ONS area page independently lists Carlisle and Copeland among Cumberland's former districts. The underlying Order also changed authorities beyond those two districts. The reader returns the immediate changes explicitly indexed for the cited code. It cannot assign an individual address to a post-change authority or infer a crosswalk solely from the scope of the legislation. The code's operational date is distinct from the founding date of the place. [ONS Cumberland record](https://www.ons.gov.uk/explore-local-statistics/areas/E06000063-cumberland) · [Structural Changes Order](https://www.legislation.gov.uk/uksi/2022/331/article/4/made)

### How was the artist credited, and for which role?

We checked two releases, one recording and its linked composition against the MusicBrainz API on November 25, 2026. The role reader needs an entity kind, MBID and exact relationship. A recording-to-work lookup lets a caller move to the composition deliberately. The release named *Cendre* bills **Fennesz + Sakamoto**. The catalog name on the latter artist's record is 坂本龍一. Its release record supplied no producer relationship. Another release, *2010LP*, bills 5Loops and independently identifies s'Poom as producer. The API gives *2010LP* a year-only release date; the snapshot retains that precision. [Cendre](https://musicbrainz.org/release/ad6c808f-2948-478a-aad5-3e02a9692e67) · [2010LP](https://musicbrainz.org/release/0e97456e-9471-4869-bf43-10ea495f666a)

The *Game Over* recording bills seven artists in order and explicitly identifies six guest vocalists. The final billed name is “Chipmunk”; the contributor's catalog name is “Chip.” The performance points to a distinct work with seven writer relationships. Neither the billing nor the six guest-vocal rows supplies the work's writer list. The work does not record a separate lyricist relationship in this capture. An empty role answer describes the selected catalog record and scope. It leaves the underlying contribution open. [Recording](https://musicbrainz.org/recording/b040d89e-190f-47a0-af7f-20534101056c) · [Work](https://musicbrainz.org/work/dee610f0-e425-311f-be39-554747afbddf)

### Replay credits through anno

The same four MusicBrainz entities can be projected into the existing Rust reader and SQLite subscriber backed by the anno CLI. The source text is a reconstruction of MusicBrainz's ordered billing and the scoped role rows. Its citation identifies the captured catalog representation. These records do not establish what a physical album cover displayed. The fixed key asks about billing, production, writing and lyrics at each entity scope. All 16 answers matched both readers. The domain reader above separately returns guest-vocal attributes and distinguishes unsupported queries from missing entities.

```sh
python native.py --verify
```

Use `--anno /path/to/anno --rust-reader /path/to/reader_answer` to reuse prebuilt executables. With no arguments, `native.py` emits the reader input without access to the answer key.

## Reproduce the capture

The offline file `sample.json` has SHA-256 `46e4848813cd61a1a33f031880c8267ed60ad14e61f0c1e6f10f2112ebfe1308`. To independently rebuild it, use a separate cache outside the checkout. The collector requests only the specified public inputs and retains the selected fields in the output. The example fetches about 89 MB of ONS archives plus four MusicBrainz records.

```sh
python capture.py --cache /tmp/anno-catalog-inputs --fetch --output /tmp/anno-catalog-rebuilt.json
cmp sample.json /tmp/anno-catalog-rebuilt.json
```

| ONS edition | SHA-256 of downloaded ZIP |
|---|---|
| [2023-12 item and product rights pointer](https://www.arcgis.com/sharing/rest/content/items/3acc892515aa49a8885c2deb734ebd3d?f=pjson) | `3dee2254ad7ccd4583b3fde832afcd7d40ae092929d5cc612fc01f3e5e727694` |
| [2025-12 item and product rights pointer](https://www.arcgis.com/sharing/rest/content/items/c59bd9b324644773b92d4f06d1f48dd7?f=pjson) | `8b913fef0dfa5637807b29eadeae3c33105d8af4e59ad2eb06389313d3437b45` |

The collector checks both ZIP digests. It sorts unordered relationship rows and preserves ordered billing and join phrases. Each MusicBrainz MBID identifies an entity whose metadata can change. The checked-in projection is a content-hashed, dated observation. A later live result can differ; retain a separate observation to study that change. These four responses provide no edit date or change rationale.

## Rights

**Source: Office for National Statistics licensed under the Open Government Licence v.3.0**. The ONS portion contains English name/code, classification, effective and termination dates, and statutory instrument identifiers from the two CSV lookup tables. The original ZIP archives, geometry, maps, postcode records and accompanying guides are outside the checked-in projection. [ONS conditions for lookups](https://www.ons.gov.uk/methodology/geography/licences) · [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)

The MusicBrainz portion consists of **CC0 core data** from the MetaBrainz Foundation and MusicBrainz contributors. It includes entity identifiers, names, dates, artist credits and relationships. No artwork, wiki text, user tags, annotations or edit history is included. The repository's code license applies to the original reader and checks. [MusicBrainz data license](https://musicbrainz.org/doc/About/Data_License) · [Distribution terms by data partition](https://musicbrainz.org/doc/MusicBrainz_Database/Download#License) · [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
