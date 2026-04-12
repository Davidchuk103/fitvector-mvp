# Техническое описание сканирования тела по фото

## Обзор流程

Система измеряет тело человека по двум фотографиям (спереди и сбоку) с использованием компьютерного зрения и методов машинного обучения.

## Этапы обработки

```
Фото (front + side)
        │
        ▼
┌───────────────────────┐
│ 1. Валидация фото    │
│    - формат (JPEG/   │
│      PNG/WebP)        │
│    - размер ≥256px   │
│    - вес ≤10MB        │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ 2. Предобработка    │
│    изображения      │
│    - CLAHE           │
│      нормализация   │
│    - контраст      │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ 3. MediaPipe Pose   │
│    определение      │
│    33 landmarks    │
│    (ключевые точки) │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ 4. Валидация позы   │
│    - руки опущены?  │
│    - ноги прямые?  │
│    - тело прямо?    │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ 5. Калибровка       │
│    масштаба          │
│    (пиксели → см)   │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ 6. Измерения        │
│    - линейные       │
│    - обхваты       │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ 7. Confidence      │
│    расчёт           │
└───────────────────────┘
        ▼
   Результат (25 замеров)
```

---

## Детали каждого этапа

### 1. Валидация фото

**Файлы:**
- `backend/measure.py:204-243` — функции `validate_photo()` и `validate_all()`

**Проверки:**
- Формат: JPEG, PNG или WebP
- Минимальный размер: 256×256 пикселей
- Максимальный вес: 10MB

**Код:**
```python
MIN_PHOTO_DIMENSION = 256
MAX_PHOTO_SIZE = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
```

---

### 2. Предобработка изображения

**Файлы:**
- `backend/measure.py:284-337` — функция `preprocess_image()`

#### 2.1 CLAHE нормализация

CLAHE (Contrast Limited Adaptive Histogram Equalization) — адаптивная нормализация гистограммы.

**Зачем нужно:**
- Разные условия освещения на разных фото
- Обеспечивает стабильнос��ь распознавания

**Код:**
```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_gray = clahe.apply(gray)
```

**Параметры:**
- `clipLimit=2.0` — ограничение контраста (предотвращает избыточное усиление шума)
- `tileGridSize=(8, 8)` — размер области адаптации (8×8 блоков)

---

### 3. MediaPipe Pose определение ключевых точек

**Используемая библиотека:** MediaPipe Pose (от Google)

**Файлы:**
- `backend/measure.py:339-402` — функция `run_mediapipe_pose()`

#### 3.1 Конфигурация MediaPipe

```python
mp_pose = mp_pose_pkg.solutions.pose.Pose(
    static_image_mode=True,    # Для фото, не видео
    model_complexity=2,         # Самая точная модель
    min_detection_confidence=0.6,  # Минимальная уверенность
)
```

- `model_complexity=2` — использует самую точную, но медленную модель
- `min_detection_confidence=0.6` — отсеивает низкокачественные распознавания
- `static_image_mode=True` — оптимизация для одиночных изображений

#### 3.2 33 Landmark точки

MediaPipe возвращает 33 точки:

| # | Точка | Описание |
|---|-------|---------|
| 0 | NOSE | Нос |
| 1 | LEFT_EYE_INNER | Левый глаз (внутр.) |
| 2 | LEFT_EYE | Левый глаз |
| 3 | LEFT_EYE_OUTER | Левый глаз (внешн.) |
| 4 | RIGHT_EYE_INNER | Правый глаз (внутр.) |
| 5 | RIGHT_EYE | Правый глаз |
| 6 | RIGHT_EYE_OUTER | Правый глаз (внешн.) |
| 7 | LEFT_EAR | Левое ухо |
| 8 | RIGHT_EAR | Правое ухо |
| 11 | LEFT_SHOULDER | Левое плечо |
| 12 | RIGHT_SHOULDER | Правое плечо |
| 13 | LEFT_ELBOW | Левый локоть |
| 14 | RIGHT_ELBOW | Правый локоть |
| 15 | LEFT_WRIST | Левое запястье |
| 16 | RIGHT_WRIST | Правое запястье |
| 23 | LEFT_HIP | Левое бедро |
| 24 | RIGHT_HIP | Правое бедро |
| 25 | LEFT_KNEE | Левое колено |
| 26 | RIGHT_KNEE | Правое колено |
| 27 | LEFT_ANKLE | Левая лодыжка |
| 28 | RIGHT_ANKLE | Правая лодыжка |

#### 3.3 Доступ к координатам

```python
def get(self, idx: int) -> Optional[tuple[float, float]]:
    p = self.points.get(idx)
    if p is None or p[2] < MIN_LANDMARK_VISIBILITY:
        return None
    return (p[0] * self.img_w, p[1] * self.img_h)
```

Возвращает `(x, y)` в пикселях. `p[2]` — visibility (видимость точки от 0 до 1).

---

### 4. Валидация позы

**Файлы:**
- `backend/measure.py:405-538` — функция `validate_pose_quality()`

Проверяет, что человек стоит правильно:

#### 4.1 Проверки

1. **Поворот тела** — нос должен быть по центру между плечами
2. **Руки подняты** — запястья должны быть ниже плеч
3. **Руки скрещены** — запястья не должны быть перед телом
4. **Ноги согнуты** — колени не должны быть смещены
5. **Стойка широкая** — лодыжки не должны быть далеко от бёдер
6. **Наклон плеч/бёдер** — плечи и бёдра должны быть горизонтально

#### 4.2 Scoring

```python
score = 1.0
if rotation_ratio > 0.15:
    issues.append("body is rotated")
    score -= 0.2
if wrist_above_shoulder:
    issues.append("arm is raised")
    score -= 0.15
# ... и так далее
```

Возвращает:
- `PoseQualityReport.is_valid` — валидна ли поза
- `PoseQualityReport.score` — оценка от 0 до 1
- `PoseQualityReport.issues` — список проблем

---

### 5. Калибровка масштаба

**Файлы:**
- `backend/measure.py:495-544` — функция `calibrate_scale()`

Самая критичная часть! Преобразует пиксели в сантиметры.

#### 5.1 Проблема

На фото разные люди могут быть разного размера. Нужен способ перевести "пиксели" в "сантиметры".

#### 5.2 Текущий метод

Используем соотношение роста человека:

```python
HEAD_RATIO_DEFAULT = 0.052  # 5.2% от роста

nose_to_ankle_cm = height_cm * HEAD_RATIO_DEFAULT
scale = nose_to_ankle_cm / nose_to_ankle_px
```

Это означает: расстояние от носа до лодыжек составляет примерно 5.2% от роста человека.

**Пример:**
- Рост: 170 см
- Ожидаемое расстояние нос-лодыжка: 170 × 0.052 = 8.84 см
- Измеренное на фото: 200 пикселей
- Масштаб: 8.84 / 200 = 0.0442 см/пиксель

#### 5.3 Дополнительная корректировка

Также пытается использовать:
- Расстояние между ушами (≈14 см)
- Межзрачковое расстояние (≈6.3 см)

```python
# Если уши видны
scale_ears = EAR_TO_EAR_AVG_CM / ear_dist_px
scale = scale_primary * 0.7 + scale_ears * 0.3
```

#### 5.4 Ограничения

**Проблемы текущего метода:**
- Предполагает, что человек стоит ровно (не наклонился)
- Предполагает полный рост виден на фото (от макушки до ступней)
- Фиксированное соотношение 5.2% может быть неточным

---

### 6. Измерения

#### 6.1 Линейные измерения

Измеряются напрямую по ключевым точкам:

```python
shoulder_w_px = front.pixel_dist(11, 12)  # Расстояние между плечами
shoulder_width = shoulder_w_px * front_scale
```

**Измеряемые значения:**
- shoulder_width — ширина плеч (Landmarks 11-12)
- shoulder_length — длина плеча (0-11)
- upper_arm_length — плечо + предплечье (11-15)
- upper_leg_length — бедро (23-25)
- full_leg_length — вся нога (23-27)
- torso_length — туловище (0-23 или 0-24)

#### 6.2 Обхваты (circumferences)

Обхваты измеряются по формуле эллипса Рамануджана:

```python
def ellipse_circumference(a: float, b: float) -> float:
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))
```

Где:
- `a` — полуширина (из фронтального фото)
- `b` — полуглубина (из бокового фото)

#### 6.3 Contour Detection

Для обхватов груди и талии используется детектирование контура тела:

**Файлы:**
- `backend/measure.py:583-613` — функция `measure_body_contour_width_at_level()`

**Методы (в порядке приоритета):**

1. **Canny Edge Detection** — поиск краёв на каждой строке
   ```python
   edges = cv2.Canny(row, 30, 100)
   ```

2. **Otsu Thresholding** — бинаризация ��зображения
   ```python
   _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   ```

3. **Canny + Dilation** — утолщение краёв
   ```python
   filled = cv2.dilate(edges, np.ones((3, 3), np.uint8))
   ```

**Логика:**
```
Для каждого y в диапазоне ±15 пикселей от уровня плеч/талии:
    1. Попробовать Canny на этой строке
    2. Если < 15px — попробовать Otsu
    3. Если всё ещё < 15px — использовать dilation
    4. Вернуть максимальную ширину
```

#### 6.4 Gender-специфичные коэффициенты

```python
MALE_BODY_RATIOS = {
    "shoulder_to_hip_ratio": 1.15,
    "chest_to_waist_ratio": 1.12,
    "waist_to_hip_ratio": 0.85,
}

FEMALE_BODY_RATIOS = {
    "shoulder_to_hip_ratio": 0.92,
    "chest_to_waist_ratio": 1.05,
    "waist_to_hip_ratio": 0.72,
}
```

Если пол известен, используются разные коэффициенты для расчёта обхватов.

---

### 7. Confidence расчёт

**Файлы:**
- `backend/measure.py:1067-1094` — функция `get_confidence()`

```python
def get_confidence(cv_detected, has_weight, pose_score=0, visibility_ratio=0, gender_known=False):
    if cv_detected:
        base = 0.75
        base += min(pose_score * 0.08, 0.08)      # До +8% за качество позы
        base += min(visibility_ratio * 0.07, 0.07) # До +7% за видимость
        if has_weight: base += 0.03              # +3% если есть вес
        if gender_known: base += 0.02             # +2% если пол известен
        return min(base, 0.95)                   # Максимум 95%
    else:
        base = 0.55
        if has_weight: base += 0.10
        return min(base, 0.65)
```

---

## Debug Info

В ответе API есть объект `debug_info` для отладки:

```json
{
  "scale_factor": 0.0442,
  "scale_method": "nose_ankle_ratio",
  "contour_method": "canny_row"
}
```

**Поля:**
- `scale_factor` — коэффициент см/пиксель
- `scale_method` — метод калибровки
  - `"nose_ankle_ratio"` — основной (нос-лодыжка)
  - `"nose_ankle_ears"` — с корректировкой по ушам
  - `"nose_ankle_eyes"` — с корректировкой по глазам
- `contour_method` — метод определения контура
  - `"canny_row"` — Canny edge detection
  - `"otsu_row"` — Otsu thresholding
  - `"canny_dilated"` — Canny + dilation

---

## Ограничения и известные проблемы

### 1. Калибровка

**Проблема:** Фиксированное соотношение 5.2% может быть неточным для разных типов телосложения.

**Решение (возможное):**
- Использовать две фотографии с известным расстоянием между предметами
- Использовать реальный рост + статистическую модель

### 2. Contour Detection

**Проблема:** На сложном фоне или при одежде похожей на фон, контур тела определяется неправильно.

**Причины:**
- Otsu thresholding чувствителен к яркости фона
- Canny может ловить границы одежды, а не тела

### 3. Perspective Distortion

**Проблема:** Если камера расположена не на уровне глаз или слишком близко/далеко, возникают искажения.

**Решение (возможное):**
- Инструкция для пользователя делать фото с определённого расстояния
- Автоматическая корректировка перспективы

### 4. Clothing

**Проблема:** Обтягивающая или свободная одежда влияет на измерения.

**Решение:**
- Инструкция надевать облегающую одежду
- Алгоритмы корректировки под одежду

---

## API Reference

### Endpoint: POST /measure

**Input (multipart/form-data):**
- `front` — фото спереди (файл)
- `side` — фото сбоку (файл)
- `height_cm` — рост в сантиметрах (число)
- `weight_kg` — вес в кг (опционально)
- `gender` — "male", "female" или "unknown"

**Response:**
```json
{
  "height_cm": 170,
  "weight_kg": 65,
  "gender": "female",
  "cv_detected": true,
  "confidence": 0.87,
  "pose_score": 0.92,
  "pose_issues": [],
  "debug_info": {
    "scale_factor": 0.0442,
    "scale_method": "nose_ankle_ratio",
    "contour_method": "canny_row"
  },
  "measurements": {
    "shoulder_width": {
      "label": "Ширина плеч",
      "value": 36.5,
      "unit": "cm",
      "method": "cv"
    },
    // ... ещё 24 измерения
  }
}
```

---

## Dependencies

```
fastapi
uvicorn
python-multipart
Pillow
mediapipe
opencv-python-headless
numpy
```

---

## Links

- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose)
- [CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_pyCLAHE.html)
- [Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
- [Ramanujan Ellipse Circumference](https://en.wikipedia.org/wiki/Ellipse#Circumference)