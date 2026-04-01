"use client";
import { useEffect } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { DetectionResult } from "./page";

// Fix Leaflet's default marker icon issue in Next.js
const icon = L.icon({
  iconUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png",
  iconRetinaUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  tooltipAnchor: [16, -28],
});

// Auto-adjust map boundaries to fit all markers dynamically
function MapBounds({ data }: { data: DetectionResult[] }) {
  const map = useMap();
  useEffect(() => {
    if (data && data.length > 0) {
      const bounds = L.latLngBounds(data.map(d => [d.lat, d.lon]));
      // Only fit bounds if they are valid
      if (bounds.isValid()) {
         map.fitBounds(bounds, { padding: [50, 50], maxZoom: 18 });
      }
    }
  }, [data, map]);
  return null;
}

export default function MapView({ data }: { data: DetectionResult[] }) {
  const center = data.length > 0 ? [data[0].lat, data[0].lon] : [45.7489, 21.2087];
  
  return (
    <MapContainer center={center as [number, number]} zoom={15} style={{ height: "100%", width: "100%", zIndex: 0 }} className="rounded-lg">
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution="&copy; OpenStreetMap contributors" />
      <MapBounds data={data} />
      {data.map((item, idx) => (
        <Marker key={`${item.cam_key}_${item.image}_${idx}`} position={[item.lat, item.lon]} icon={icon}>
          <Popup>
            <div className="font-mono text-xs">
              <strong>Conf:</strong> {(item.conf * 100).toFixed(0)}% <br/>
              <strong>Image:</strong> {item.image} <br/>
              <strong>Cam:</strong> {item.cam_key}
            </div>
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
}
