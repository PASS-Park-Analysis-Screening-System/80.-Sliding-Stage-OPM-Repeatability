# OPM Repeatability 지표 알고리즘 명세

> 연구소 기술 문의 답변 (2026-04) 기반 정리.
> 기준 Tool 소스 코드 확인 결과를 반영한 정확한 계산 방식입니다.

---

## 1. 공통 전처리

### 1-1. Order-1 LS Flatten (Edge-Only Fitting)

모든 지표는 동일한 Flatten을 사용합니다.

| 항목 | 값 |
|------|-----|
| 방식 | 1차 최소자승 (LS) 회귀 |
| Fitting 구간 | **양끝 1% 픽셀만** (left 1% + right 1%) |
| 적용 구간 | 전체 픽셀 |

**수식:**
```
edge_pixels = max(1, int(N * 0.01))
x_fit = x[:edge_pixels] + x[-edge_pixels:]     # 양끝 1%만
z_fit = z[:edge_pixels] + z[-edge_pixels:]

coefficients = polyfit(x_fit, z_fit, order=1)   # 1차 LS 회귀
regression = polyval(coefficients, x_all)       # 전체 구간에 평가

z_flat = z_raw - regression                     # 전체에서 차감
```

**핵심**: 기존 `FlattenProcessor.flatten(edge_percent=1.0)`은 inner pixels로 fitting (양끝 제외).
연구소 기준은 **양끝 pixels로만 fitting** (정반대). `edge_only_flatten()` 함수로 구현.

### 1-2. Outlier Pixel Exclusion

분석 전에 비정상 픽셀을 제외합니다.

**기준**: 각 pixel 위치에서 repeat간 Max-Min 범위 계산 → 범위가 큰 pixel 제외

| 모드 | 설명 |
|------|------|
| Percentile | 상위 x% pixel 제외 (예: 5% → pixel_range 상위 5% 제외) |
| Pixels | 상위 N개 pixel 제외 (예: 10 → range가 가장 큰 10개 pixel 제외) |
| None | 제외 없음 (모든 pixel 사용) |

**수식:**
```
stack = [z_flat_repeat1, z_flat_repeat2, ..., z_flat_repeatN]  # shape: (repeats, pixels)
pixel_range = stack.max(axis=0) - stack.min(axis=0)             # shape: (pixels,)

# Percentile 모드 (예: value=5.0)
threshold = percentile(pixel_range, 95)     # 100 - 5 = 95번째 백분위
valid_mask = pixel_range <= threshold       # 상위 5% 제외

# Pixels 모드 (예: value=10)
sorted_idx = argsort(pixel_range)
valid_mask[sorted_idx[-10:]] = False        # 상위 10개 제외
```

---

## 2. 지표별 계산 공식

### 2-1. Rep. Max (Repeatability Maximum)

**의미**: 동일 위치에서 repeat간 높이 편차의 최대값

```
pixel_range[i] = max(z_flat[repeat, i]) - min(z_flat[repeat, i])   # 각 유효 pixel i
Rep. Max = max(pixel_range[valid_pixels])
```

### 2-2. Rep. 1σ (Repeatability 1-Sigma)

**의미**: 각 pixel에서 repeat간 표준편차의 RMS 집계

```
pixel_std[i] = std(z_flat[:, i], ddof=1)    # 각 유효 pixel i에서 repeat간 표본표준편차
Rep. 1σ = sqrt(mean(pixel_std[valid_pixels]²))   # RMS 집계
```

**주의**: `std(pixel_range)`가 아님. pixel별 repeat std → RMS.

**ddof=1 (표본표준편차)**: repeat 수 N=5에서 불편추정량이며 기준 Tool과 정합. `ddof=0`(모표준편차)은
재현성을 약 12% 과소평가하여 낙관적 판정(기준이 FAIL할 것을 PASS) 위험이 있어 사용하지 않음.
검증: ddof=0 → 기준 대비 −14%, ddof=1 → −3.6% (data/25mm 4개 Position 평균).

### 2-3. OPM Max (Optical Profiler Measurement Maximum)

**의미**: 각 repeat에서 유효 pixel의 높이 범위(Max-Min)의 최대값

```
OPM[r] = max(z_flat[r, valid]) - min(z_flat[r, valid])    # repeat r의 유효 pixel OPM
OPM Max = max(OPM[r])                                      # 전체 repeat 중 최대
```

### 2-4. OPM 1σ

**의미**: Leveling 후 프로파일 형상의 RMS 크기 (Bow의 RMS)

```
all_heights = z_flat[all_repeats, valid_pixels].ravel()    # 모든 repeat × 유효 pixel
OPM 1σ = sqrt(mean(all_heights²))                          # RMS from zero
```

**핵심**: 이것은 5회 OPM의 변동성(반복성)이 **아닙니다**.
모든 repeat의 유효 pixel 높이값 전체에 대한 zero 기준 RMS입니다.
평균을 다시 빼지 않으므로 "from zero"입니다.

---

## 3. 검증 기준값 (1mm 데이터, 1_LT Position)

| 지표 | 기준 Tool | 이전 구현 (v1) | 수정 후 목표 |
|------|-----------|---------------|-------------|
| Rep. Max | 7.596 nm | 7.595 nm (✅) | ~7.596 nm |
| Rep. 1σ | 1.736 nm | 1.096 nm (❌ -37%) | ~1.736 nm |
| OPM Max | 104.788 nm | 102.231 nm (⚠️) | ~104.788 nm |
| OPM 1σ | 67.801 nm | 1.810 nm (❌ 37.6배) | ~67.801 nm |

### 이상치 사례: 3_RT Position, Sample18

| 항목 | 값 | 비고 |
|------|-----|------|
| Raw Range | 148.160 nm | 정상 범위 97~100 nm 대비 +50% |
| Rep. Max (이상치 포함) | 62.083 nm | 기준 Tool 9.834 nm 대비 6.3배 |
| OPM Max (이상치 포함) | 149.397 nm | 기준 Tool 104.412 nm 대비 +43% |
| OPM Max (Sample18 제외) | ~104.540 nm | 기준 Tool과 근접 |

→ Outlier 모드 활성화 시 Sample18의 비정상 pixel이 자동 제외되어야 함.

---

## 4. 이전 구현과의 차이 (v1 → v2)

| 항목 | v1 (이전) | v2 (수정) |
|------|-----------|-----------|
| Rep 용 Flatten | Order-2, 전체구간 LS | **Order-1, 양끝 1% fitting** |
| OPM 용 Flatten | Order-1, 전체구간 LS | **Order-1, 양끝 1% fitting** |
| Outlier 제외 | 없음 | **pixel Max-Min 기반 상위 x%/N개 제외** |
| Rep. 1σ 공식 | `std(pixel_range)` | **`sqrt(mean(pixel_stds²))`** — RMS |
| OPM 1σ 공식 | `std(opm_values)` — 5회 OPM의 stdev | **`sqrt(mean(all_heights²))`** — RMS from zero |
| OPM 계산 pixel | 전체 pixel | **유효 pixel만** |

---

## 5. 기본값 및 검증 한계

### 5-1. Outlier 처리 — 기본 Raw, "이상치 제외값" 병기(선택)
참 측정값을 숨기지 않기 위해 기본은 **Raw(전체 데이터, 이상치 미제외)**만 Summary에 표시하고, 사용자가
Summary 탭의 **"이상치 제외값 병기"** 체크 시 각 지표 칸 하단에 **이상치 제외값**(반복 간 편차가 큰 이상
픽셀 제외, 기본 Percentile 1%)을 함께 표시한다(두 값 상이 시 노랑 강조).
- Raw가 headline(차트/상세/판정 기본). 이상치 제외값은 **체크 시에만 계산**(`current_result_robust`).
  이상치 제외 **임계값(Mode/Value)** 조정은 Admin 전용(Summary 탭 헤더), 병기 토글은 전원 가능.
- 이상치 = 픽셀 위치별 repeat **Max−Min**이 큰 픽셀. Summary 탭 **ⓘ** 버튼이 현재 데이터에서 제외 대상
  픽셀을 빨강으로 시각화(`plot_manager.create_outlier_illustration_figure`).
- QC-5(Median±3·MAD)가 이상치를 독립 WARN → 은폐 없음. Raw/이상치 제외값 판정 불일치 시 Spec 패널이 경고.
- 효과: 불량 repeat(3_RT/Sample18 148nm) 시 Raw RepMax **62.8** / 이상치 제외값 **9.4**가 나란히 보여
  차이의 원인(이상 픽셀)을 근거와 함께 제시 가능. (내부 식별자/상수는 `ROBUST_OUTLIER_MODE/VALUE` 유지)
- Percentile 1%는 완만·scale-invariant 기본값(정책 선택) — 레거시 AFP threshold 재현이 목적 아님.

### 5-2. OPM Max 잔차 (~4~5%, intrinsic)
OPM Max는 기준 Tool 값이 **제공된 TIFF의 raw peak-to-valley보다도 높아**(예: 1_LT 기준 104.788 > raw 101.2)
어떤 leveling으로도 재현 불가하다(두 데이터셋 확증, z_scale=1.0로 단순 스케일 버그 아님). OPM 1σ 등은 정합하므로
알고리즘 오류가 아니라 2021 AFP Tool의 DAC→nm 변환/데이터 인스턴스 차이로 판단하며, 닫으려면 AFP 변환식이 필요하다.
회귀 점검은 `scripts/validate_against_reference.py`로 수행(OPM Max는 intrinsic으로 분류).

## 6. 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|-----------|
| v1.0 | 2026-03 | 초기 구현 (Order-2 flatten for Rep, Order-1 for OPM, 전체구간 fitting) |
| v2.0 | 2026-04 | 연구소 답변 기반 수정_PMS: Q&A 4249 (Order-1 edge-only flatten, outlier exclusion, 공식 변경) |
| v2.1 | 2026-06 | 최종 검수: Rep.1σ ddof=1(표본), Outlier Raw+Robust 병기 표시, 표시/상세 leveling 일치, 검증 하니스 추가 |
| v2.2 | 2026-06 | 도구 기능(판정 알고리즘 불변): 일반/Admin 접근제어(PIN), Spec/Recipe 프리셋(`analyze_recipe(spec_overrides=)`로 내장 SPEC_* range별 override), MSA/Gauge R&R(반복성 EV Type-1), PDF 검수 리포트, Summary 2줄 병기. 상세 PRD §17 |
| v2.3 | 2026-06 | UX/IA 정리(알고리즘 불변): Summary 기본 Raw + **"이상치 제외값" 병기 토글**(기본 OFF)·outlier 컨트롤 Summary 탭 이관·셀 읽기전용·값 스핀박스 수정; Spec 관리/QC/Compare/MSA = **Quality(Admin 전용)**, 일반=고정 표준 Spec; **"Robust"→"이상치 제외값"** 표기 통일; 이상치 **ⓘ 시각화 팝업**(제외 픽셀 빨강) |
