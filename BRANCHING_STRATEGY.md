# Branching Strategy for EGW Query Expansion System Remediation

## Overview
This document defines the branching strategy, merge policies, and rollback procedures for remediation work on the EGW Query Expansion System, ensuring systematic and traceable changes with comprehensive testing checkpoints.

## Branch Naming Conventions

### Primary Branch Types

#### 1. Remediation Branches
- **Format**: `remediation/{type}-{issue-id}-{short-description}`
- **Examples**:
  - `remediation/security-001-dependency-vulnerabilities`
  - `remediation/performance-002-memory-optimization`
  - `remediation/stability-003-pipeline-error-handling`
  - `remediation/compliance-004-code-quality-standards`

#### 2. Emergency Hotfix Branches  
- **Format**: `hotfix/emergency-{severity}-{issue-id}-{short-description}`
- **Examples**:
  - `hotfix/emergency-critical-001-pipeline-crash`
  - `hotfix/emergency-high-002-memory-leak`

#### 3. Feature Enhancement Branches
- **Format**: `feature/enhancement-{component}-{short-description}`
- **Examples**:
  - `feature/enhancement-retrieval-egw-optimization`
  - `feature/enhancement-monitoring-metrics-collection`

#### 4. Testing and Validation Branches
- **Format**: `test/{remediation-branch-name}-validation`
- **Examples**:
  - `test/remediation-security-001-validation`
  - `test/performance-baseline-validation`

### Branch Hierarchy and Relationships

```
main (production-ready)
├── develop (integration branch)
├── staging (pre-production validation)
├── remediation/*
│   ├── remediation/security-*
│   ├── remediation/performance-*
│   ├── remediation/stability-*
│   └── remediation/compliance-*
├── hotfix/emergency-*
├── feature/enhancement-*
└── test/*-validation
```

## Merge Policies and Review Requirements

### 1. Code Review Requirements

#### Mandatory Reviewers by Branch Type:
- **Remediation branches**: Minimum 2 reviewers (1 technical lead, 1 domain expert)
- **Emergency hotfix**: Minimum 1 senior reviewer + post-merge review
- **Feature enhancement**: Minimum 2 reviewers including architecture review
- **Testing validation**: Minimum 1 reviewer with testing expertise

#### Review Checklist:
- [ ] Code follows established patterns and conventions
- [ ] Security implications assessed and addressed
- [ ] Performance impact evaluated
- [ ] Documentation updated (README, architecture docs)
- [ ] Breaking changes identified and communicated
- [ ] Backward compatibility maintained or migration path provided

### 2. Automated Testing Checkpoints

#### Pre-merge Requirements:
```bash
# All branches must pass:
1. Static Analysis: ruff check egw_query_expansion/
2. Type Checking: mypy egw_query_expansion/
3. Unit Tests: pytest egw_query_expansion/tests/ --cov=90
4. Integration Tests: pytest tests/integration/ -v
5. Performance Regression Tests: python scripts/performance_monitor.py --baseline-check
6. Security Scan: bandit -r egw_query_expansion/
```

#### Branch-Specific Additional Requirements:

**Remediation Branches**:
```bash
# Security remediation
- CVE scan: safety check
- Dependency audit: pip-audit

# Performance remediation  
- Memory profiling: python -m memory_profiler scripts/memory_test.py
- Performance benchmarks: python scripts/performance_monitor.py --full-suite

# Stability remediation
- Fault injection tests: python tests/fault_injection_suite.py
- Load testing: python tests/load_test_suite.py
```

**Emergency Hotfix Branches**:
```bash
# Fast-track validation (must complete in <30 minutes)
- Critical path tests only
- Smoke tests: python tests/smoke_tests.py
- Production simulation: python tests/production_simulation.py
```

### 3. Merge Strategy

#### Standard Merge Process:
1. **Squash and merge** for feature branches (clean history)
2. **Merge commit** for remediation branches (preserve remediation context)
3. **Rebase and merge** for hotfix branches (linear history)

#### Merge Commands:
```bash
# Remediation branch merge (preserve history)
git checkout develop
git merge --no-ff remediation/security-001-dependency-vulnerabilities
git tag -a "remediation-security-001" -m "Security remediation: dependency vulnerabilities"

# Feature branch merge (squash)
git checkout develop  
git merge --squash feature/enhancement-retrieval-egw-optimization
git commit -m "feat: EGW retrieval optimization

- Implemented advanced caching layer
- Optimized memory usage by 30%
- Added performance monitoring hooks

Closes: #ENH-001"

# Hotfix merge (rebase)
git checkout main
git merge --ff-only hotfix/emergency-critical-001-pipeline-crash
git tag -a "hotfix-critical-001" -m "Emergency fix: pipeline crash resolution"
```

## Rollback Procedures

### 1. Rollback Classification

#### Level 1: Simple Rollback (Single Commit)
```bash
# Identify problematic commit
git log --oneline -10

# Create rollback branch
git checkout -b rollback/revert-commit-{hash}

# Revert specific commit
git revert {commit-hash}
git push origin rollback/revert-commit-{hash}

# Create PR for rollback merge
```

#### Level 2: Feature Rollback (Multiple Commits)
```bash
# Identify range of commits to rollback
git log --oneline --since="2024-01-01" --grep="feature-name"

# Create rollback branch
git checkout -b rollback/revert-feature-{feature-name}

# Revert commit range (oldest to newest)
git revert {oldest-hash}^..{newest-hash}

# Resolve conflicts if any
git status
git add .
git commit -m "fix: resolve rollback conflicts"

git push origin rollback/revert-feature-{feature-name}
```

#### Level 3: Large-Scale Remediation Rollback
```bash
# For major remediation rollbacks affecting multiple components

# 1. Create emergency rollback branch
git checkout -b emergency-rollback/remediation-{type}-{id}

# 2. Identify all affected commits (use tags)
git log --oneline $(git merge-base main HEAD)..HEAD

# 3. Create comprehensive revert
git revert --mainline 1 {merge-commit-hash}

# 4. Validate rollback doesn't break dependencies
python scripts/dependency_check.py --validate-rollback

# 5. Run emergency test suite
python tests/emergency_validation.py

# 6. If tests pass, fast-track merge to main
git checkout main
git merge --ff-only emergency-rollback/remediation-{type}-{id}

# 7. Deploy rollback
git tag -a "emergency-rollback-$(date +%Y%m%d-%H%M)" -m "Emergency rollback: {reason}"
```

### 2. Rollback Validation Procedure

#### Pre-Rollback Checklist:
- [ ] Impact assessment completed
- [ ] Stakeholder notification sent  
- [ ] Backup of current state created
- [ ] Rollback branch created and tested
- [ ] Emergency contact list activated

#### Post-Rollback Validation:
```bash
# 1. System health check
python scripts/health_check.py --full-system

# 2. Performance validation
python scripts/performance_monitor.py --baseline-compare

# 3. Integration validation  
pytest tests/integration/ --rollback-validation

# 4. User acceptance testing
python tests/user_acceptance_suite.py

# 5. Monitor for 24 hours
python scripts/monitoring_dashboard.py --alert-mode
```

### 3. Emergency Rollback Procedures

#### Critical System Failure (< 5 minutes response time)
```bash
# Immediate automated rollback
git checkout main
git reset --hard HEAD~1  # If last commit is problematic
git push --force-with-lease origin main

# OR use pre-tested rollback branch
git checkout main
git merge emergency-rollback-prepared
git push origin main

# Immediate deployment
./deploy.sh --emergency --skip-tests
```

#### High Priority Issues (< 30 minutes response time)
```bash
# Create targeted rollback
git checkout -b emergency-rollback-$(date +%Y%m%d-%H%M)

# Selective revert of specific changes
git revert {problematic-commits} --no-commit
git commit -m "emergency: rollback problematic changes

Reason: {specific-issue}
Impact: {affected-components}
Validation: {quick-tests-run}"

# Fast-track merge
git checkout main
git merge --ff-only emergency-rollback-$(date +%Y%m%d-%H%M)
git push origin main
```

## Branch Protection Rules

### Main Branch Protection:
- Require pull request reviews (minimum 2 approvals)
- Require status checks to pass before merging
- Require branches to be up to date before merging  
- Include administrators in restrictions
- Allow force pushes only for emergency rollbacks with approval

### Develop Branch Protection:
- Require pull request reviews (minimum 1 approval)
- Require status checks to pass before merging
- Allow administrators to bypass requirements for hotfixes

### Remediation Branch Guidelines:
- Must be created from latest `develop` branch
- Regular rebase with `develop` to prevent conflicts
- Must include performance impact assessment
- Required to pass full test suite including new regression tests

## Monitoring and Compliance

### Branch Monitoring:
```bash
# Daily branch health check
python scripts/branch_monitor.py --check-all

# Weekly cleanup of stale branches  
python scripts/branch_cleanup.py --dry-run

# Monthly remediation effectiveness report
python scripts/remediation_report.py --generate-monthly
```

### Compliance Tracking:
- All remediation branches must link to tracking issues
- Merge commits must include impact assessment
- Rollback procedures must be documented and tested quarterly
- Branch naming compliance checked via pre-commit hooks

This branching strategy ensures systematic, traceable, and reversible remediation work while maintaining system stability and enabling rapid response to critical issues.