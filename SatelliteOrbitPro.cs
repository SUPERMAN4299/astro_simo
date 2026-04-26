using UnityEngine;

public class SatelliteOrbitPro : MonoBehaviour
{
    [Header("References")]
    public Transform earth;

    [Header("Orbit Settings")]
    public float orbitSpeed = 20f;
    public float orbitHeight = 3f;
    public float inclination = 0f; // 0 for GEO

    private float angle = 0f;
    private float orbitRadius; // LOCKED radius

    void Start()
    {
        if (earth != null)
        {
            Renderer renderer = earth.GetComponent<Renderer>();

            if (renderer != null)
            {
                float earthRadius = renderer.bounds.extents.magnitude;
                orbitRadius = earthRadius + orbitHeight;
            }
        }
    }

    void Update()
    {
        if (earth == null) return;

        // Stable angular motion
        angle += orbitSpeed * Time.deltaTime;

        float rad = angle * Mathf.Deg2Rad;

        Vector3 orbit = new Vector3(
            Mathf.Cos(rad) * orbitRadius,
            0,
            Mathf.Sin(rad) * orbitRadius
        );

        // Apply inclination (0 = GEO)
        orbit = Quaternion.Euler(inclination, 0f, 0f) * orbit;

        transform.position = earth.position + orbit;
    }
}