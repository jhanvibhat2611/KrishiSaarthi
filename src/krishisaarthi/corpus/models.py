"""Review records are separate from immutable Phase 1 source records."""
from datetime import datetime
from typing import Literal
from pydantic import Field, model_validator
from ..schemas import Record, EvidenceSpan


class RegistryEntry(Record):
    document_id: str = Field(min_length=1)
    path: str
    content_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    title: str
    publication_type: str
    publisher: str
    publication_date: str = Field(pattern=r'^\d{4}(-(0[1-9]|1[0-2]))?$')
    date_precision: Literal['year', 'month'] = 'month'
    geographic_applicability: list[str] = Field(default_factory=list)
    copyright_restrictions: str | None = None
    # Limits audit/proposals only; canonical extraction still preserves every page.
    audit_page_numbers: list[int] | None = None
    metadata_evidence_pages: list[int]
    source_url: str | None = None
    publisher_homepage: str | None = None
    source_languages: list[str]
    license_status: Literal['unknown', 'verified'] = 'unknown'
    license_evidence: str | None = None
    license_verified_by: str | None = None
    redistribution_permission: Literal['not_established', 'verified'] = 'not_established'
    printed_page_labels: dict[str, str | None] = Field(default_factory=dict)
    # Source-specific findings from inspection, not general automatic fact correction.
    known_issue_anchors: list[dict[str, str | int]] = Field(default_factory=list)

    @model_validator(mode='after')
    def licensing(self):
        if (len(self.publication_date) == 4) != (self.date_precision == 'year'):
            raise ValueError('Publication date must match date precision')
        if self.audit_page_numbers is not None and (
            not self.audit_page_numbers or self.audit_page_numbers != sorted(set(self.audit_page_numbers))
            or any(p < 1 for p in self.audit_page_numbers)
        ):
            raise ValueError('Audit pages must be nonempty, positive, sorted and unique')
        if self.license_status == 'verified' and not (self.license_evidence and self.license_verified_by):
            raise ValueError('Verified license requires evidence and reviewer')
        if self.redistribution_permission == 'verified' and self.license_status != 'verified':
            raise ValueError('Redistribution cannot be inferred from accessibility')
        return self


class Issue(Record):
    issue_id: str
    code: str
    document_id: str
    physical_page: int
    start_char: int
    end_char: int
    detail: str
    # Automatic warnings require explicit reviewer resolution before admission.
    blocking: bool = True


class ReviewChecks(Record):
    english: bool = False
    wheat_relevant: bool = False
    privacy_clear: bool = False
    extraction_verified: bool = False
    duplicate_clear: bool = False
    consistency_clear: bool = False
    table_clear: bool = False
    printed_label_verified: bool = False


class ReviewedRegion(Record):
    region_id: str
    evidence_span: EvidenceSpan
    physical_page: int = Field(ge=1)
    printed_page_label: str | None = None
    section_boundaries: list[int] = Field(default_factory=list)
    proposed_section: str | None = None
    recommendation: Literal['candidate', 'hold']
    status: Literal['pending', 'approved', 'excluded'] = 'pending'
    issue_ids: list[str] = Field(default_factory=list)
    checks: ReviewChecks = Field(default_factory=ReviewChecks)
    reviewer: str | None = None
    reviewed_at: str | None = None
    review_note: str | None = None
    # Map issue ID to the reviewer's substantive resolution. Not auto-populated.
    resolutions: dict[str, str] = Field(default_factory=dict)
    reviewed_content_fingerprint: str | None = None

    @model_validator(mode='after')
    def approval(self):
        a, b = self.evidence_span.start_char, self.evidence_span.end_char
        if self.section_boundaries != sorted(set(self.section_boundaries)):
            raise ValueError('Section boundaries must be unique and sorted')
        if any(not a < point < b for point in self.section_boundaries):
            raise ValueError('Section boundaries must lie strictly inside the region')
        if self.status == 'approved':
            if not all(self.checks.model_dump(exclude={'schema_version'}).values()):
                raise ValueError('Approval requires every manual review check')
            if not all(x and x.strip() for x in (self.reviewer, self.reviewed_at, self.review_note, self.reviewed_content_fingerprint)):
                raise ValueError('Approval requires reviewer, timestamp, note and fingerprint')
            if datetime.fromisoformat(self.reviewed_at).tzinfo is None:
                raise ValueError('Review timestamp must include a timezone')
        return self
