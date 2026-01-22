# Feature Dictionary (High-level)

The dataset contains 30 numeric features computed from digitized images of fine needle aspirate (FNA) of breast masses.
They are typically grouped into three families for each underlying measurement:
- **mean**: average value
- **se**: standard error
- **worst**: “worst” (largest) value

## Common measurements (intuitive meaning)
- **radius**: average distance from center to points on the perimeter
- **texture**: standard deviation of gray-scale values
- **perimeter**: perimeter length
- **area**: area size
- **smoothness**: local variation in radius lengths
- **compactness**: (perimeter² / area) - 1.0
- **concavity**: severity of concave portions of the contour
- **concave points**: number of concave portions
- **symmetry**: symmetry measure
- **fractal dimension**: “coastline approximation” - 1

## Feature list (as provided by the dataset)
- `mean radius`
- `mean texture`
- `mean perimeter`
- `mean area`
- `mean smoothness`
- `mean compactness`
- `mean concavity`
- `mean concave points`
- `mean symmetry`
- `mean fractal dimension`
- `radius error`
- `texture error`
- `perimeter error`
- `area error`
- `smoothness error`
- `compactness error`
- `concavity error`
- `concave points error`
- `symmetry error`
- `fractal dimension error`
- `worst radius`
- `worst texture`
- `worst perimeter`
- `worst area`
- `worst smoothness`
- `worst compactness`
- `worst concavity`
- `worst concave points`
- `worst symmetry`
- `worst fractal dimension`
