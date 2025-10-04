import React, { useState, useEffect, useCallback, type JSX } from 'react';

// --- Configuration ---
// IMPORTANT: Replace this with the public URL you get from ngrok when you run your Colab server
const API_BASE_URL = "https://nondeaf-unhortatively-olinda.ngrok-free.dev";

// --- TypeScript Interfaces ---
interface InvoiceData {
  company_name: string | null;
  invoice_number: string | null;
  date: string | null;
  total: string | null;
}

interface SubmissionStatus {
  type: 'success' | 'error';
  message: string;
}

// --- Helper Icon Components ---
const IconUpload: React.FC = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
  </svg>
);
const IconEdit: React.FC = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" /><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
  </svg>
);
const IconSave: React.FC = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" /><polyline points="17 21 17 13 7 13 7 21" /><polyline points="7 3 7 8 15 8" />
    </svg>
);

// --- Main Application Component ---
export default function App(): JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // Data states for our workflow
  const [imageId, setImageId] = useState<string | null>(null);
  const [aiPrediction, setAiPrediction] = useState<InvoiceData | null>(null);
  const [userCorrection, setUserCorrection] = useState<InvoiceData | null>(null);
  
  // UI state
  const [isEditing, setIsEditing] = useState<boolean>(false);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [submissionStatus, setSubmissionStatus] = useState<SubmissionStatus | null>(null);

  // Function to reset all states for a new upload
  const resetAllStates = () => {
    setFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setIsLoading(false);
    setError(null);
    setImageId(null);
    setAiPrediction(null);
    setUserCorrection(null);
    setIsEditing(false);
    setIsSubmitting(false);
    setSubmissionStatus(null);
  };

  // Handle new file selection from user
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      resetAllStates();
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
    }
  };

  // --- API Call 1: Process Invoice ---
  const processInvoice = useCallback(async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/process_invoice/`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
      
      const data = await response.json();
      setImageId(data.image_id);
      setAiPrediction(data.extracted_data);
      setUserCorrection(data.extracted_data); // Initialize form with AI data
    } catch (err) {
      setError('Failed to process invoice. Is the server running and the API URL correct?');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  }, [file]);
  
  // Automatically call processInvoice when a new file is set
  useEffect(() => {
    if (file && !imageId) {
      processInvoice();
    }
  }, [file, imageId, processInvoice]);

  // Update form state as user types
  const handleCorrectionChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setUserCorrection(prev => (prev ? { ...prev, [name]: value } : null));
  };

  // --- API Call 2: Log Correction ---
  const handleSaveCorrection = async () => {
    if (!imageId || !aiPrediction || !userCorrection) return;
    
    setIsSubmitting(true);
    setSubmissionStatus(null);
    
    const payload = {
      image_id: imageId,
      ai_prediction: aiPrediction,
      user_correction: userCorrection
    };

    try {
      const response = await fetch(`${API_BASE_URL}/log_correction/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(`Server error: ${response.statusText}`);

      const result = await response.json();
      if (result.status === 'success') {
        setSubmissionStatus({ type: 'success', message: 'Correction saved! Thank you.' });
        setIsEditing(false); // Lock the form after successful save
      } else {
        throw new Error(result.message || 'Failed to save correction.');
      }
    } catch (err) {
      setSubmissionStatus({ type: 'error', message: 'Failed to save. Please try again.' });
      console.error(err);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="bg-slate-50 min-h-screen font-sans text-slate-800 flex flex-col items-center p-4 sm:p-6">
      <div className="w-full max-w-md mx-auto">
        <header className="text-center my-8">
          <h1 className="text-3xl font-bold text-slate-900 tracking-tight">SmartBill AI</h1>
          <p className="text-slate-600 mt-2">Upload an invoice to begin.</p>
        </header>

        <main className="bg-white p-6 rounded-2xl shadow-lg transition-all">
          {!file ? (
            <div className="flex flex-col items-center justify-center p-8 border-2 border-dashed border-slate-300 rounded-xl">
              <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center text-center text-slate-500 hover:text-blue-600 transition-colors duration-300">
                <IconUpload />
                <span className="mt-2 font-semibold">Click to upload</span>
                <p className="text-sm mt-1">or take a photo</p>
              </label>
              <input id="file-upload" type="file" className="hidden" accept="image/*" capture="environment" onChange={handleFileChange} />
            </div>
          ) : (
            <div>
              <div className="relative mb-4 group">
                {previewUrl && <img src={previewUrl} alt="Invoice preview" className="rounded-xl w-full h-auto object-contain max-h-[50vh]" />}
                {isLoading && (
                  <div className="absolute inset-0 bg-white/80 backdrop-blur-sm flex flex-col items-center justify-center rounded-xl">
                    <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                    <p className="mt-4 font-semibold text-slate-700">Analyzing Invoice...</p>
                  </div>
                )}
              </div>

              {error && <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded-lg mb-4" role="alert">{error}</div>}
              
              {userCorrection && (
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <h2 className="text-xl font-bold">Scan & Verify</h2>
                    {!isEditing && !submissionStatus && (
                      <button onClick={() => setIsEditing(true)} className="flex items-center gap-2 px-3 py-2 text-sm font-semibold text-blue-600 bg-blue-100 rounded-lg hover:bg-blue-200 transition-colors">
                        <IconEdit /> Correct
                      </button>
                    )}
                  </div>

                  <div className="space-y-3">
                    {(Object.keys(userCorrection) as Array<keyof InvoiceData>).map((key) => (
                      <div key={key}>
                        <label className="text-sm font-medium text-slate-500 capitalize">{key.replace(/_/g, ' ')}</label>
                        <input
                          type="text"
                          name={key}
                          value={userCorrection[key] || ''}
                          onChange={handleCorrectionChange}
                          readOnly={!isEditing}
                          className={`mt-1 block w-full px-3 py-2 bg-slate-50 border rounded-md shadow-sm text-slate-900 ${isEditing ? 'border-blue-300 ring ring-blue-200' : 'border-slate-200'}`}
                        />
                      </div>
                    ))}
                  </div>

                  {submissionStatus && (
                    <div className={`p-3 rounded-lg text-sm font-semibold ${submissionStatus.type === 'success' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                      {submissionStatus.message}
                    </div>
                  )}

                  {isEditing && (
                    <div className="flex flex-col sm:flex-row gap-3 pt-2">
                      <button onClick={handleSaveCorrection} disabled={isSubmitting} className="w-full flex justify-center items-center gap-2 bg-blue-600 text-white font-bold py-2.5 px-4 rounded-lg hover:bg-blue-700 transition-colors disabled:bg-blue-300">
                        {isSubmitting ? <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div> : <IconSave />}
                        {isSubmitting ? 'Saving...' : 'Save Correction'}
                      </button>
                      <button onClick={() => { setIsEditing(false); setUserCorrection(aiPrediction); setSubmissionStatus(null); }} className="w-full bg-slate-200 text-slate-800 font-bold py-2.5 px-4 rounded-lg hover:bg-slate-300 transition-colors">
                        Cancel
                      </button>
                    </div>
                  )}
                </div>
              )}
              
              <button onClick={resetAllStates} className="w-full text-center mt-6 text-sm font-medium text-slate-500 hover:text-slate-800 transition-colors">
                Scan another invoice
              </button>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

