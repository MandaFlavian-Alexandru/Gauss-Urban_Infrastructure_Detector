# Spatial Mathematics Specification

The core technical differentiator of the Gauss Firida Detector pipeline is its spatial projection engine. This engine transforms pixel-level bounding boxes detected in raw imagery into physical 3D ground coordinates relative to the vehicle, and finally mapping them into absolute WGS84 coordinates.

## 1. Equidistant Fisheye Ray-Casting Projection

Ladybug 5 sensors utilize ultra-wide lenses that do not map linearly. To resolve the true angle of deviation (\(\theta\)) from the optical center, we employ an Equidistant Fisheye Lens model characterized by the equation:

\[ r = f \cdot \theta \]

Where:
* \(r\) is the absolute pixel distance from the optical center to the target point within the frame.
* \(f\) is the estimated focal length.
* \(\theta\) is the true incident angle of the ray in radians departing the optical center.

First, we establish the optical center and calculate \(f\) using the hardware's known Horizontal Field of View (\(FOV_H = 60^\circ\)). The extreme horizontal edge of the sensor represents a \(30^\circ\) deviation:
\[ f = \frac{\text{Center}_x}{\text{radians}(FOV_H / 2)} \]

By determining the actual offset coordinates of the bounding box \((dx, dy)\), we find \(r = \sqrt{dx^2 + dy^2}\) and subsequently the deviation angle \( \theta = r / f \). We then project this as a 3D unit vector extending from the camera into physical space.

## 2. 2D-to-3D Ground Plane Intersection

The camera array is mounted at a known physical height, strictly defined as \(Y = 2.45m\).
Assuming localized level terrain, the pipeline calculates the true geometric ground intersection by extending the 3D visual ray until it reaches the ground plane. The intersection scalar \(t\) is derived from the downward \(Y\) vector:

\[ t = \frac{\text{CAMERA\_HEIGHT}}{\text{Ray}_y} \]

Applying this scalar to the lateral and forward vectors establishes the specific physical \(X\) and \(Z\) coordinates on the terrain relative to the optical center. The localized angular deviation of the discovered point from the camera's baseline orientation is extracted via simple trigonometry:

\[ \alpha = \text{degrees}(\arctan(X / Z)) \]

## 3. World Space Transformation

The absolute geographic bearing incorporates three variables:
1. The vehicle's heading (extracted via GPS telemetry).
2. The specific camera's fixed horizontal mount offset (e.g., Camera 3 is rotated \(60^\circ\) relative to chassis forward).
3. The localized intersection deviation angle (\(\alpha\)) calculated above.

This computes a true absolute bearing. A standard spherical earth formulation (assuming \(R_{earth} = 6378137.0m\)) processes the bearing and the ground distance against the telemetry's baseline coordinate, extracting the final absolute WGS84 Latitude and Longitude.

## 4. Haversine Spatial Deduplication & Clustering

Cross-camera overlaps and consecutive time-series frames cause extreme data redundancy for static infrastructure objects. A single distribution box may be observed a dozen times as the vehicle passes. To mitigate this, the backend processes spatial proximities using the Haversine formula:

\[ a = \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos\phi_1\cdot\cos\phi_2\cdot\sin^2\left(\frac{\Delta\lambda}{2}\right) \]
\[ c = 2 \cdot \text{atan2}\left(\sqrt{a}, \sqrt{1-a}\right) \]
\[ d = R \cdot c \]

### Filtering Criteria
* **5.0m Deduplication Zone**: If two signals calculate coordinates within 5.0 meters of each other, they are considered the exact same object. A confidence-based election ensures only the bounding box yielding the highest inference confidence score is retained; the weaker signal is permanently discarded.
* **2.0m Proximity Clustering Warning**: If separate, structurally deduplicated targets end up residing within 2.0 meters of each other, they are mathematically flagged as `CLUSTERED`. This indicates dense anomalies or conflicting geometric split boundaries, explicitly warning downstream engineers that the targets require manual review.
