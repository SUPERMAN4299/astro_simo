using UnityEngine;

public class SatelliteOrbitPro : MonoBehaviour
{
    [Header("References")]
    public Transform earth;

    [Header("Orbit Settings")]
    public float orbitSpeed = 20f;
    public float orbitHeight = 2f;     // Height above Earth surface
    public float inclination = 45f;    // Orbit tilt angle

    private float angle = 0f;
    private float earthRadius;

    void Start()
    {
        if (earth != null)
        {
            Renderer renderer = earth.GetComponent<Renderer>();

            if (renderer != null)
            {
                // Real Earth size from mesh
                earthRadius = renderer.bounds.extents.x;
            }
        }
    }

    void Update()
    {
        if (earth == null) return;

        // Increase angle over time
        angle += orbitSpeed * Time.deltaTime;

        float rad = angle * Mathf.Deg2Rad;

        // Total orbit radius
        float radius = earthRadius + orbitHeight;

        // Base circular orbit (flat)
        Vector3 orbit = new Vector3(
            Mathf.Cos(rad) * radius,
            0,
            Mathf.Sin(rad) * radius
        );

        // Apply inclination (tilt orbit plane)
        orbit = Quaternion.Euler(inclination, 0f, 0f) * orbit;

        // Final position
        transform.position = earth.position + orbit;

        // Optional: Make satellite face forward along orbit
        transform.LookAt(earth);
    }
}