# AeroSense ChartQA Dataset

## Dataset Summary

Synthetic aviation chart Q&A pairs for instruction fine-tuning of language models on
aeronautical knowledge — chart symbology, approach procedures, airspace, navigation aids,
and FAA regulations.

Generated using Claude claude-sonnet-4-20250514 from public FAA AIM, TERPS criteria, and
Jeppesen chart specification documentation.

## Format

Alpaca instruction-tuning format (JSONL):

```json
{
  "instruction": "What does a dashed magenta circle on a VFR sectional chart indicate?",
  "input": "",
  "output": "A dashed magenta circle indicates Class E airspace that extends from the surface...",
  "category": "chart_symbology",
  "difficulty": "intermediate",
  "safety_critical": false,
  "source_reference": "FAA AIM 3-2-6"
}
```

## Categories

| Category | Count | Description |
|----------|-------|-------------|
| chart_symbology | ~80 | VFR/IFR chart symbols, colors, annotations |
| approach_procedures | ~80 | IAP plates, ILS, RNAV, missed approach |
| airspace | ~80 | Class A–G, TFRs, special use airspace |
| navigation_aids | ~80 | VOR, GPS, ILS, NDB, DME |
| faa_regulations | ~80 | FARs, AIM references, compliance |
| safety_critical | ~100 | Safety-critical questions requiring caution |

## Source

Synthetically generated from public FAA documentation:
- FAA Aeronautical Information Manual (AIM)
- Title 14 CFR (FARs)
- FAA TERPS (Terminal Instrument Procedures) criteria
- FAA Chart User's Guide

## License

CC BY 4.0 — Attribution required. Not for use in actual flight operations without
verification against current official FAA publications.
