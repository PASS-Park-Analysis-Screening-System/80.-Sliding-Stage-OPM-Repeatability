# Scan Parameter 표준 명칭 정의서 (PASS OPM 공통)

> **목적** — Park Systems 프로파일 TIFF에서 추출하는 스캔/측정 파라미터의 **명칭·단위·출처·추출 가능성**을
> 프로젝트 간 동일하게 표기하기 위한 표준. 모든 PASS OPM 계열 툴(80/70/97 등)은 이 표를 단일 기준으로 사용한다.
> 근거: `src/core/tiff_reader.py` (태그/offset 파서) + 실측 파일 `25mm_..._0001_Height.tiff` 검증.

## 0. 범례

**출처(Source)**
- `H@NNN` — 바이너리 헤더(TIFF 태그 **50435**, 580 byte)의 byte offset `NNN`
- `RAW` — 원시 프로파일 데이터(TIFF 태그 **50434**, float32 배열)
- `XML` — 확장 헤더(TIFF 태그 **50441**, UTF-8 XML) — *현재 코드 미파싱*
- `TIFF` — 표준 TIFF 태그
- `DERIVED` — 다른 값에서 계산
- `FOLDER` — 폴더/파일명 구조 (TIFF 내부 아님)

**추출 가능성(Extract)**
- ✅ 신뢰성 있게 추출 (값 검증됨)
- △ 부분/주의 (필드는 있으나 의미 제한·값 비신뢰·모달리티 상이)
- ❌ 파일에 없음 (헤더·XML 어디에도 미내장)

---

## 1. 현재 패널 표기 항목 (Scan Parameters 패널) — 표준 확정

| 표준 명칭 | 표시 라벨 | 단위 | 물리적 의미 | 출처 | Extract | 실측값(예) |
|---|---|---|---|---|---|---|
| `recipe` | Recipe | — | 측정 레시피/폴더명 | FOLDER | ✅ | Profile_25mm_Dynamic |
| `scan_size` | Size | µm | 스캔 이송 길이(1D 프로파일 전체 길이) | H@140 (f64) | ✅ | 25000 |
| `pixel_count` | Px | px | 프로파일 픽셀 수 = RAW 길이 | RAW len | ✅ | 8192 |
| `resolution` | Resolution | nm/px | 픽셀당 물리 크기 = Size×1000 ÷ Px | DERIVED | ✅ | 3051.8 |
| `scan_speed` | Speed | mm/s | 스테이지 이송 속도 | H@172 (f64) | ✅ | 0.100 |
| `set_point` | SP | (모드별) | Set Point — 피드백 제어 기준값 | H@180 (f64) | ✅ | 30.0 |
| `z_servo_gain` | Z Gain | — | Z 서보(피드백) 게인 | H@284 (f64) | ✅ | 1.5 |

> Resolution 검산: 25000 µm × 1000 ÷ 8192 px = **3051.76 nm/px**.

---

## 2. 추가로 추출 가능한 항목 (현재 패널 미노출 — 차기 노출 후보)

| 표준 명칭 | 단위 | 의미 | 출처 | Extract | 실측값(예) |
|---|---|---|---|---|---|
| `source` | — | 신호 채널 (Height / Z Drive) | H@4 (UTF-16) | ✅ | Height |
| `z_sensitivity` | m (또는 nm/V) | Z 감도 — DAC→nm 환산 계수. 부호 보존 | H@220 (f64) | ✅ | −6.413e-9 |
| `z_scale` | — | Z 스케일 계수 | H@228 (f64) | ✅ | 1.0 |
| `head_mode` | — | 헤드/유닛 식별자 (컨트롤러 타입) | H@68 (UTF-16) | △ | C-AFM |
| `xy_offset_x/y` | — | XY 오프셋 | H@156 / H@164 | △ | *상수 정의됨, 파서 미수록* |
| `datetime` | — | 측정 일시 | TIFF 306 | ✅ | 2025:07:31 17:58:13 |
| `wafer_diameter` | mm | 웨이퍼 직경 (DieMapInfo) | XML | ✅ | 300.0 |
| `die_coord_x/y` | µm | 스테이지/다이 좌표 | XML | ✅ | 59999.5 / 235004.9 |
| `tip_mileage` | µm | 팁 누적 사용 거리 | XML | ✅ | 1350000 |
| `tip_approach_count` | — | 팁 어프로치 횟수 | XML | ✅ | 55 |
| `sample_rotation` | ° | 시료 회전각 | XML | ✅ | 0.0 |

> **Z 환산식**: `Z(nm) = raw / 2^20 × |z_sensitivity(m)| × 1e9` (DAC 20-bit). `tiff_reader.py:12-14`.

---

## 3. 스크린샷 "테스트 조건" 항목 추출 판정 (2D AFM 이미지 조건표)

> ⚠️ **전제** — 그 조건표는 **2D AFM 이미지**(80µm · 256×256 · 0.8Hz · AC160 · Contact) 측정 기준이고,
> 이 툴이 읽는 데이터는 **1D OPM 프로파일**(25mm · 8192px)이다. **측정 모달리티가 다르다.**
> 아래는 *1D 프로파일 TIFF*에서의 추출 판정이다.

| 조건표 항목 | 값(예) | Extract | 근거 |
|---|---|---|---|
| Source | Height (Forward) | ✅ | H@4 `source='Height'` (+파일명 보조) |
| Scan size | 80 µm | ✅ | H@140 (프로파일은 25000µm) |
| Scan Rate | 0.8 Hz | △ | 속도(mm/s)만 저장, **Hz는 미저장** (2D 라인레이트 개념) |
| Head mode | Contact | △ | H@68 필드 존재하나 값이 `'C-AFM'`(컨트롤러), "Contact" 아님 |
| Scan pixel | 256×256 | △ | 프로파일은 **1D 8192px**; 256×256은 렌더 이미지(TIFF 256·257) |
| Sample | Optical Flat | △ | XML `<SampleName>` 존재하나 값이 `'System'`(기본/비신뢰) |
| **Cantilever** | AC160 | ❌ | XML TipInfo엔 포트/슬롯/마일리지만, **모델명 없음** |
| **Over scan** | 10 % | ❌ | 헤더·XML 미내장 |
| **XY servo mode** | On | ❌ | Z 서보 게인만 존재, XY 서보 On/Off 없음 |
| **Fast Scan Axis** | X/Y | ❌ | 1D 프로파일은 단일 축 — 미저장 |
| **XY / Z Range mode** | 1 / 1 | ❌ | 헤더·XML 미내장 |

### 판정
**Raw 프로파일 TIFF만으로 그 조건표를 완전 복원하는 것은 불가능하다.**
- 약 절반(Cantilever·Over scan·XY servo·Fast Scan Axis·Range mode)은 파일에 **아예 없다**.
- 일부(Scan Rate·Scan pixel·Fast Scan Axis)는 **2D 이미징 전용 개념**으로 1D 프로파일엔 대응값이 없거나 다르다.
- 반대로 프로파일 TIFF에는 조건표에 **없는** 계측 핵심값(Z Sensitivity, Z Servo Gain, Set Point,
  Z Scale, 웨이퍼 die 좌표, Tip mileage)이 들어 있어, *프로파일 측정의 표준 메타데이터는 §1–§2로 충분히 정의된다.*

---

## 4. 비고
- **50435 헤더 미해독 영역** — 580 byte 중 현재 디코드는 offset ≤ 284뿐. offset 188–219, 236–283,
  292–579 구간은 미파싱이며, 추가 파라미터가 더 있을 수 있다(역공학 여지).
- **50441 XML 미파싱** — 현재 `read_profile()`은 XML(50441)을 읽지 않는다. §2의 XML 항목을 쓰려면
  파서 추가 필요(차기 결정 사항).
- 표준 명칭(`snake_case`)은 코드 필드명 기준, 표시 라벨은 패널/리포트 기준. 신규 프로젝트는 이 표를 복사해
  동일 명칭·단위·라벨을 사용한다.
