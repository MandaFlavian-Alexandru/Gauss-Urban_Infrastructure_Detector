"use client";

import { useState, useRef } from "react";
import Image from "next/image";
import dynamic from "next/dynamic";

export interface DetectionResult {
  image: string;
  cam_key: string;
  folder_path?: string;
  x1?: number;
  y1?: number;
  x2?: number;
  y2?: number;
  conf: number;
  lat: number;
  lon: number;
  clustered?: boolean;
}

// Dynamically import MapView to prevent SSR issues with Leaflet
const MapView = dynamic(() => import("./MapView"), { ssr: false, loading: () => <div className="h-full w-full bg-gray-100 animate-pulse flex items-center justify-center text-gray-400 font-mono text-xs">Loading OpenStreetMap...</div> });

export default function Home() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [cameraProgress, setCameraProgress] = useState({Camera1: 0, Camera2: 0, Camera3: 0, Camera4: 0});
  const [isComplete, setIsComplete] = useState(false);
  const [terminalLog, setTerminalLog] = useState<string[]>([]);
  
  // Calculate average of 4 cameras for true overall progress
  const trueOverallProgress = Math.round(
    (cameraProgress.Camera1 + cameraProgress.Camera2 + cameraProgress.Camera3 + cameraProgress.Camera4) / 4
  );

  // Real Integration State
  const [folderPath, setFolderPath] = useState("");
  const [minConfidence, setMinConfidence] = useState(75);
  const [parallaxRadius, setParallaxRadius] = useState(5.00);
  const [isRulesExpanded, setIsRulesExpanded] = useState(false);
  
  // Results State
  const [resultsData, setResultsData] = useState<DetectionResult[]>([]);

  // Analytics State
  const [startTime, setStartTime] = useState<number>(0);
  const [processDuration, setProcessDuration] = useState<string>("0m 0s");

  // Lightbox State
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [zoomLevel, setZoomLevel] = useState<number>(1);

  const pollInterval = useRef<NodeJS.Timeout | null>(null);

  // Lightbox Handlers
  const handleOpenLightbox = (index: number) => {
    setSelectedIndex(index);
    setZoomLevel(1);
  };
  const handleCloseLightbox = () => {
    setSelectedIndex(null);
    setZoomLevel(1);
  };
  const handlePrevImage = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedIndex(prev => (prev !== null && prev > 0) ? prev - 1 : resultsData.length - 1);
    setZoomLevel(1);
  };
  const handleNextImage = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedIndex(prev => (prev !== null && prev < resultsData.length - 1) ? prev + 1 : 0);
    setZoomLevel(1);
  };
  const handleZoomIn = (e: React.MouseEvent) => {
    e.stopPropagation();
    setZoomLevel(prev => Math.min(prev + 0.5, 4));
  };
  const handleZoomOut = (e: React.MouseEvent) => {
    e.stopPropagation();
    setZoomLevel(prev => Math.max(prev - 0.5, 0.5));
  };
  const handleWheel = (e: React.WheelEvent) => {
    if (e.deltaY < 0) {
      setZoomLevel(prev => Math.min(prev + 0.2, 5));
    } else {
      setZoomLevel(prev => Math.max(prev - 0.2, 0.5));
    }
  };

  const handleStartAnalysis = async () => {
    if (!folderPath) {
      alert("Please enter a valid Trident network/local path.");
      return;
    }
    
    setIsProcessing(true);
    setProgress(0);
    setTerminalLog(["Initializing API connection to local node..."]);
    setIsComplete(false);
    setResultsData([]);
    const currentStartTime = Date.now();
    setStartTime(currentStartTime);
    
    try {
      const res = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          folder_path: folderPath,
          min_confidence: minConfidence,
          cluster_radius: parallaxRadius
        })
      });

      if (!res.ok) {
        throw new Error("API not reachable");
      }

      pollInterval.current = setInterval(async () => {
        try {
          const statusRes = await fetch("http://localhost:8000/api/status");
          const data = await statusRes.json();
          
          setProgress(data.progress);
          setCameraProgress(data.camera_progress || {Camera1: 0, Camera2: 0, Camera3: 0, Camera4: 0});
          setTerminalLog(data.logs);
          
          if (!data.is_running && data.results_ready) {
            if (pollInterval.current) clearInterval(pollInterval.current);
            
            // Fetch compiled analysis payload from inference node
            const resData = await fetch("http://localhost:8000/api/results");
            const finalResults = await resData.json();

            // Calculate precise duration using captured scope variable to avoid React closure traps
            const durationMs = Date.now() - currentStartTime;
            const m = Math.floor(durationMs / 60000);
            const s = Math.floor((durationMs % 60000) / 1000);
            setProcessDuration(`${m}m ${s}s`);
            
            // Validate detection payload and handle empty edge cases
            if (!Array.isArray(finalResults) || finalResults.length === 0) {
                 setTimeout(() => {
                     setIsProcessing(false);
                     alert("No Firidas (Urban Infrastructure instances) were detected in this strict recording. Adjust rules or select a new recording.");
                 }, 500);
            } else {
                 setResultsData(finalResults);
                 setTimeout(() => {
                   setIsProcessing(false);
                   setIsComplete(true);
                 }, 1000);
            }
          }
        } catch (e) {
          console.error("Polling error:", e);
        }
      }, 1000);

    } catch (e) {
      console.error("Failed to start analysis:", e);
      setTerminalLog(["ERROR: Could not connect to Gauss Python Node on Port 8000."]);
      setTimeout(() => setIsProcessing(false), 3000);
    }
  };

  const handleCancelAnalysis = async () => {
    try {
      await fetch("http://localhost:8000/api/cancel", { method: "POST" });
      if (pollInterval.current) clearInterval(pollInterval.current);
      setIsProcessing(false);
      setProgress(0);
      setCameraProgress({Camera1: 0, Camera2: 0, Camera3: 0, Camera4: 0});
    } catch (e) {
      console.error(e);
    }
  };

  const handleDelete = async (image_name: string, cam_key: string) => {
    // Optimistic UI updates
    const newResults = resultsData.filter(d => !(d.image === image_name && d.cam_key === cam_key));
    setResultsData(newResults);
    
    try {
      await fetch("http://localhost:8000/api/delete_result", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_name, cam_key })
      });
    } catch (e) {
      console.error("Failed to sync deletion with backend:", e);
    }
  };


  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm relative z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Image src="/gauss_logo.png" alt="Gauss Logo" width={140} height={40} className="object-contain" priority/>
            <div className="h-8 w-px bg-gray-300 hidden sm:block"></div>
            <h1 className="text-xl font-bold text-gray-700 tracking-tight hidden sm:block">
              Urban Infrastructure Detector
            </h1>
          </div>
          <div className="text-sm border border-brand-primary text-brand-primary px-3 py-1 rounded-full font-semibold uppercase tracking-wider">
            Enterprise Node Active
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full relative z-10">
        {!isComplete ? (
          <div className="max-w-3xl mx-auto space-y-6">
            {/* Input View (Control Panel) */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
              <div className="p-6 border-b border-gray-100">
                <h2 className="text-lg font-bold text-gray-900 mb-4">Select Trident Recording</h2>
                
                {/* TEXT INPUT FOR ABSOLUTE PATH */}
                <div className="bg-gray-50 flex flex-col gap-2 p-6 rounded-lg border border-gray-200">
                  <label className="text-sm font-semibold text-gray-700">Root Directory Path</label>
                  <input 
                    type="text" 
                    value={folderPath}
                    onChange={(e) => setFolderPath(e.target.value)}
                    placeholder="e.g. \\srvNAS5\DEER2024\..."
                    className="w-full px-4 py-3 border border-gray-300 rounded focus:border-brand-primary focus:ring-1 focus:ring-brand-primary outline-none text-sm font-mono shadow-inner"
                  />
                  <p className="text-xs text-gray-500 mt-1">Provide the absolute local or network path containing the Trident camera subfolders.</p>
                </div>
              </div>

              {/* Advanced Rules */}
              <div className="border-b border-gray-100">
                <button 
                  onClick={() => setIsRulesExpanded(!isRulesExpanded)}
                  className="w-full px-6 py-4 flex items-center justify-between bg-gray-50 hover:bg-gray-100 transition-colors outline-none"
                >
                  <span className="text-sm font-semibold text-gray-700">Advanced Rules</span>
                  <svg className={`w-5 h-5 text-gray-500 transform transition-transform ${isRulesExpanded ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
                {isRulesExpanded && (
                  <div className="px-6 py-4 space-y-4 bg-white animate-fade-in">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-700">Minimum Confidence (%)</span>
                      <input 
                        type="number" 
                        value={minConfidence}
                        onChange={(e) => setMinConfidence(parseInt(e.target.value) || 0)}
                        className="w-24 px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:border-brand-primary focus:ring-1 focus:ring-brand-primary outline-none text-center font-mono" 
                        min="0"
                        max="100"
                        step="1"
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-700">Parallax Dedupe Radius (m)</span>
                      <input 
                        type="number" 
                        value={parallaxRadius.toFixed(2)}
                        onChange={(e) => setParallaxRadius(parseFloat(e.target.value))}
                        className="w-24 px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:border-brand-primary focus:ring-1 focus:ring-brand-primary outline-none text-center font-mono" 
                        step="0.01"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Action */}
              <div className="p-6 bg-gray-50 flex flex-col gap-3">
                <button
                  onClick={handleStartAnalysis}
                  disabled={isProcessing}
                  className={`w-full py-4 px-6 rounded-full font-bold uppercase tracking-wide transition-all shadow-md flex items-center justify-center gap-2 ${
                    isProcessing 
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed shadow-none' 
                      : 'bg-brand-primary text-white hover:bg-brand-secondary hover:shadow-lg hover:-translate-y-0.5'
                  }`}
                >
                  {isProcessing && (
                     <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-gray-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                       <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                       <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                     </svg>
                  )}
                  {isProcessing ? 'Analysis in Progress...' : 'Start New AI Analysis'}
                </button>
                
                {/* Back & Forth Toggle */}
                {resultsData.length > 0 && !isProcessing && (
                  <button 
                    onClick={() => setIsComplete(true)}
                    className="w-full py-3 px-6 rounded-full font-bold uppercase text-sm tracking-wide border-2 border-brand-primary text-brand-primary hover:bg-brand-primary hover:text-white transition-colors flex items-center justify-center gap-2 shadow-sm"
                  >
                    View Last Master Map ({resultsData.length} records)
                    <svg className="w-4 h-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
                  </button>
                )}
              </div>
            </div>

            {/* Live Dashboard (Processing View) */}
            {isProcessing && (
              <div className="bg-gray-900 rounded-xl shadow-lg border border-gray-700 p-8 text-white animate-fade-in">
                <div className="mb-8 flex flex-col sm:flex-row sm:justify-between sm:items-start gap-4">
                  <h3 className="text-lg font-semibold text-gray-300 uppercase tracking-widest pr-4">
                    Processing recording: <span className="text-white">{folderPath.match(/Record\((\d+)\)/i)?.[1] || "Unknown"}</span>
                  </h3>
                  <div className="flex flex-col sm:items-end gap-3">
                    <span className="text-brand-accent font-mono font-bold text-2xl">{trueOverallProgress}% OVERALL</span>
                    <button 
                      onClick={handleCancelAnalysis}
                      className="bg-red-600 hover:bg-red-500 text-white px-4 py-1.5 rounded-full text-xs font-bold tracking-wider uppercase flex items-center gap-2 shadow-lg transition-colors w-full sm:w-auto justify-center"
                    >
                      <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M6 18L18 6M6 6l12 12"/></svg>
                      Cancel Processing
                    </button>
                  </div>
                </div>
                
                {/* 4 Big Camera Progress Bars */}
                <div className="flex flex-col gap-6">
                  {['Camera1', 'Camera2', 'Camera3', 'Camera4'].map(cam => (
                    <div key={cam} className="bg-black/40 p-5 rounded-xl border border-gray-800 shadow-inner">
                      <div className="flex justify-between text-sm text-gray-300 mb-3 font-mono uppercase tracking-widest">
                        <span className="font-bold flex items-center gap-2">
                           <svg className={`w-4 h-4 ${cameraProgress[cam as keyof typeof cameraProgress] === 100 ? 'text-green-400' : 'text-gray-500 animate-pulse'}`} fill="currentColor" viewBox="0 0 20 20"><circle cx="10" cy="10" r="10"/></svg>
                           {cam}
                        </span>
                        <span className="text-white font-bold">{cameraProgress[cam as keyof typeof cameraProgress]}%</span>
                      </div>
                      <div className="w-full bg-gray-800 rounded-full h-4 overflow-hidden shadow-inner border border-gray-700">
                        <div 
                          className="bg-brand-accent h-4 rounded-full transition-all duration-500 ease-out relative"
                          style={{ width: `${cameraProgress[cam as keyof typeof cameraProgress]}%` }}
                        >
                           <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Results View (Deliverables) */
          <div className="animate-fade-in h-full flex flex-col gap-6 w-full fade-in">
            <div className="flex flex-col sm:flex-row sm:justify-between sm:items-end gap-4">
              <div>
                <h2 className="text-2xl font-black text-gray-900 tracking-tight">Analysis Complete</h2>
                <div className="flex flex-col sm:flex-row items-baseline gap-3 mt-1">
                  <p className="text-sm text-gray-500">Identified {resultsData.length} valid Firide utility infrastructure instances.</p>
                  <span className="text-xs font-mono font-bold bg-brand-primary/10 text-brand-primary px-3 py-1 rounded-full border border-brand-primary/20 shadow-sm flex items-center gap-1.5 whitespace-nowrap">
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Processed in {processDuration}
                  </span>
                </div>
              </div>
              <button 
                onClick={() => setIsComplete(false)}
                className="text-sm text-brand-primary hover:text-brand-secondary hover:underline font-semibold tracking-wide flex items-center gap-1"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                Configure New Analysis
              </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Left Side: Dynamic Gallery */}
              <div className="lg:col-span-2 bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex flex-col overflow-hidden max-h-[75vh]">
                <div className="flex justify-between items-center mb-4 border-b border-gray-100 pb-3">
                   <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wider">Detected Object Postages (Live)</h3>
                   <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded font-mono border border-gray-200">Sort: Internal AI</span>
                </div>
                
                <div className="flex-1 overflow-y-auto min-h-0 pr-2 pb-4 custom-scrollbar">
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {resultsData.length === 0 ? (
                    <div className="col-span-full py-12 flex flex-col items-center justify-center text-gray-400">
                       <svg className="w-12 h-12 mb-3 opacity-20" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
                       <p>No valid detections or empty API payload.</p>
                    </div>
                  ) : (
                    resultsData.map((item, idx) => (
                      <div 
                        key={`${item.cam_key}_${item.image}_${idx}`} 
                        className={`border rounded-xl overflow-hidden relative group shadow-sm transition-all hover:shadow-md cursor-pointer ${item.clustered ? 'border-yellow-400 ring-1 ring-yellow-400/50 shadow-yellow-100' : 'border-gray-200'}`}
                        onClick={() => handleOpenLightbox(idx)}
                      >
                        {/* Confidence score indicator */}
                        <div className={`absolute top-2 left-2 text-white text-[10px] font-bold px-2 py-0.5 rounded shadow z-10 flex items-center gap-1 ${item.clustered ? 'bg-yellow-500' : (item.conf >= 0.85 ? 'bg-green-500' : 'bg-orange-500')}`}>
                          {item.clustered ? "CLUSTERED" : `${(item.conf * 100).toFixed(0)}% CONF`}
                        </div>
                        
                        {/* False positive removal handler */}
                        <button 
                          onClick={(e) => { e.stopPropagation(); handleDelete(item.image, item.cam_key); }}
                          className="absolute top-2 right-2 bg-red-500 text-white rounded p-1.5 shadow-lg opacity-0 group-hover:opacity-100 hover:bg-red-600 transform transition-all hover:scale-110 z-20"
                          title="Remove False Positive (Updates Shapefile)"
                        >
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                             <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>

                        <div className="aspect-square bg-gray-100 relative overflow-hidden">
                          {/* Mount dynamic image proxy from analysis server */}
                          <Image src={`http://localhost:8000/images/${item.cam_key}_${item.image}`} alt={item.image} fill className="object-cover transition-transform duration-700 ease-in-out group-hover:scale-110 unoptimized" unoptimized/>
                        </div>
                        <div className={`p-3 text-[10px] font-mono border-t flex flex-col gap-0.5 ${item.clustered ? 'bg-yellow-50 text-yellow-800 border-yellow-200' : 'bg-gray-50 text-gray-500 border-gray-200'}`}>
                          <span className="truncate w-full font-bold" title={item.image}>{item.image}</span>
                          <span className="opacity-70">{item.cam_key}</span>
                        </div>
                      </div>
                    ))
                  )}
                  </div>
                </div>
              </div>

              {/* Right Side: Map & Download */}
              <div className="lg:col-span-1 space-y-6 flex flex-col h-full">
                <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex flex-col h-[400px]">
                  <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wider mb-4 border-b border-gray-100 pb-3 flex justify-between items-center">
                    Real Spatial Context
                    <svg className="w-4 h-4 text-brand-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </h3>
                  <div className="flex-1 rounded-lg overflow-hidden relative shadow-inner z-0">
                     <MapView data={resultsData} />
                  </div>
                </div>

                <div className="bg-brand-primary p-6 rounded-xl shadow-lg relative overflow-hidden group">
                  <div className="absolute -top-10 -right-10 w-32 h-32 bg-white opacity-5 rounded-full blur-2xl group-hover:scale-150 transition-transform duration-1000"></div>
                  <h3 className="text-white font-bold mb-2 text-lg">Master Shapefile Ready</h3>
                  <p className="text-brand-light/90 text-sm mb-6 leading-relaxed">Local Master Map contains {resultsData.length} valid entities.</p>
                  <a 
                    href="http://localhost:8000/api/download_shapefile"
                    download="Master_Firide_Map.zip"
                    target="_blank"
                    className="w-full bg-white text-brand-primary py-3.5 rounded-lg font-bold text-sm tracking-widest uppercase hover:shadow-xl hover:bg-gray-50 transform transition-all group-hover:-translate-y-1 flex items-center justify-center gap-3 cursor-pointer"
                  >
                    <svg className="w-5 h-5 text-brand-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    Download .shp
                  </a>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Lightbox Modal */}
      {selectedIndex !== null && resultsData[selectedIndex] && (
        <div className="fixed inset-0 z-[100] bg-black/90 flex items-center justify-center p-4 sm:p-8 animate-fade-in" onClick={handleCloseLightbox}>
          {/* Close Button */}
          <button className="absolute top-4 right-4 text-white bg-black/50 hover:bg-black/80 rounded-full p-2 z-[110] transition-colors" onClick={handleCloseLightbox}>
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
          </button>

          {/* Controls Footer */}
          <div className="absolute bottom-8 left-1/2 -translate-x-1/2 bg-black/60 backdrop-blur rounded-full px-6 py-3 flex items-center gap-6 z-[110] text-white shadow-xl border border-white/10" onClick={(e) => e.stopPropagation()}>
            <button onClick={handlePrevImage} className="hover:text-brand-accent transition-colors">
              <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" /></svg>
            </button>
            <div className="w-px h-6 bg-white/20"></div>
            <button onClick={handleZoomOut} className="hover:text-brand-accent transition-colors">
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7" /></svg>
            </button>
            <span className="font-mono font-bold text-sm w-12 text-center">{Math.round(zoomLevel * 100)}%</span>
            <button onClick={handleZoomIn} className="hover:text-brand-accent transition-colors">
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" /></svg>
            </button>
            <div className="w-px h-6 bg-white/20"></div>
            <button onClick={handleNextImage} className="hover:text-brand-accent transition-colors">
              <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" /></svg>
            </button>
          </div>

          {/* Image Container */}
          <div className="w-full h-full flex flex-col items-center justify-center overflow-auto custom-scrollbar" onClick={(e) => e.stopPropagation()} onWheel={handleWheel}>
            <div className="relative transition-transform duration-200 ease-out flex-shrink-0" style={{ transform: `scale(${zoomLevel})` }}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img 
                src={`http://localhost:8000/images/${resultsData[selectedIndex].cam_key}_${resultsData[selectedIndex].image}`} 
                alt={resultsData[selectedIndex].image}
                className="max-w-[90vw] max-h-[85vh] object-contain rounded shadow-2xl"
                draggable={false}
              />
            </div>
            {/* Meta overlay */}
            <div className="absolute top-4 left-4 bg-black/60 backdrop-blur px-4 py-2 rounded-lg text-white font-mono text-sm border border-white/10 z-[110] flex flex-col gap-2">
              <div>
                <span className="text-brand-accent font-bold">[{selectedIndex + 1} / {resultsData.length}]</span><br/>
                File: {resultsData[selectedIndex].image}<br/>
                Camera: {resultsData[selectedIndex].cam_key}<br/>
                Conf: {(resultsData[selectedIndex].conf * 100).toFixed(0)}%
              </div>
              <button 
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(resultsData[selectedIndex].image, resultsData[selectedIndex].cam_key);
                  // Update lightbox state index gracefully
                  if (resultsData.length <= 1) {
                    handleCloseLightbox();
                  } else if (selectedIndex === resultsData.length - 1) {
                    setSelectedIndex(selectedIndex - 1);
                  }
                }}
                className="bg-red-600 hover:bg-red-500 text-white px-3 py-1.5 rounded text-xs font-bold uppercase flex items-center justify-center gap-1.5 transition-colors w-full shadow border border-red-500/30"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                Delete Target
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
