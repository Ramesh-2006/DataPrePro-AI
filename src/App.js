import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';

const AiBotAssistantImage = 'https://img.icons8.com/?size=100&id=YFbzdUk7Q3F8&format=png&color=000000';
const GoogleSignInIcon = 'https://upload.wikimedia.org/wikipedia/commons/4/4a/Logo_2013_Google.png';

const BACKEND_URL = "http://127.0.0.1:8000"; // Ensure this matches your FastAPI port

function App() {
    const [currentPage, setCurrentPage] = useState('home');
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [sessionId, setSessionId] = useState(null);
    const [userInfo, setUserInfo] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [modal, setModal] = useState({ isVisible: false, title: '', message: '', isError: false });
    const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
    const [aiChatHistory, setAiChatHistory] = useState([]);
    const [aiChatInput, setAiChatInput] = useState('');
    const [dfInfo, setDfInfo] = useState(null);
    const [fileInfo, setFileInfo] = useState(null);
    const [cleaningRecommendations, setCleaningRecommendations] = useState([]);
    const [cleaningSteps, setCleaningSteps] = useState([]);
    const [generatedPlot, setGeneratedPlot] = useState(null);
    const [activeExploreTab, setActiveExploreTab] = useState('preview');
    const [activeCleanTab, setActiveCleanTab] = useState('recommendations');

    const aiMessagesRef = useRef(null);
    const hasSessionWarningShown = useRef(false);
    const hasFetchedRecommendationsOnce = useRef(false); // Correctly defined as a ref

    // Load session and user info from localStorage on component mount
    useEffect(() => {
        const storedSessionId = localStorage.getItem('session_id');
        const storedUserInfo = localStorage.getItem('user_info');
        if (storedSessionId) {
            setSessionId(storedSessionId);
            setIsLoggedIn(true);
            if (storedUserInfo) {
                try {
                    setUserInfo(JSON.parse(storedUserInfo));
                } catch (e) {
                    console.error('Failed to parse user info from localStorage', e);
                }
            }
        }

        const loadGoogleApi = () => {
            if (window.gapi && window.gapi.auth2) {
                window.gapi.auth2.init({
                    client_id: "742645900363-64sksrhkn5qe8a6fmaepq59la9papn1t.apps.googleusercontent.com",
                    scope: 'profile email'
                }).then(() => {
                    console.log('Google Auth2 initialized successfully.');
                }).catch(error => {
                    console.error('Failed to initialize Google Auth2:', error);
                });
            } else {
                setTimeout(loadGoogleApi, 100);
            }
        };

        if (!document.querySelector('script[src="https://apis.google.com/js/platform.js"]')) {
            const script = document.createElement('script');
            script.src = 'https://apis.google.com/js/platform.js';
            script.async = true;
            script.defer = true;
            script.onload = loadGoogleApi;
            script.onerror = () => console.error('Failed to load Google Platform script.');
            document.body.appendChild(script);
        } else {
            loadGoogleApi();
        }
    }, []);

    // Scroll chat to bottom when messages update
    useEffect(() => {
        if (aiMessagesRef.current) {
            aiMessagesRef.current.scrollTop = aiMessagesRef.current.scrollHeight;
        }
    }, [aiChatHistory]);

    // Function to change the active page
    const navigateTo = useCallback((page) => {
        setCurrentPage(page);
    }, []);

    // Modal functions
    const showModal = useCallback((title, message, isError = false) => {
        setModal({ isVisible: true, title, message, isError });
    }, []);

    const closeModal = useCallback(() => {
        setModal({ isVisible: false, title: '', message: '', isError: false });
    }, []);

    // Google Sign-In handler
    const handleGoogleSignInClick = useCallback(async () => {
        if (!window.gapi || !window.gapi.auth2) {
            showModal("Error", "Google API not loaded. Please try refreshing the page.", true);
            return;
        }
        setIsLoading(true);
        try {
            const auth2 = window.gapi.auth2.getAuthInstance();
            const googleUser = await auth2.signIn();
            const idToken = googleUser.getAuthResponse().id_token;
            const profile = googleUser.getBasicProfile();

            const res = await axios.post(`${BACKEND_URL}/api/google-auth`, { token: idToken });
            const data = res.data;
            setSessionId(data.session_id);
            setUserInfo(data.user);
            setIsLoggedIn(true);
            localStorage.setItem('session_id', data.session_id);
            localStorage.setItem('user_info', JSON.stringify(data.user));
            showModal("Welcome", `Signed in as ${data.user.name || data.user.email}`);
            navigateTo('upload');
        } catch (error) {
            showModal("Sign-In Error", error.response?.data?.detail || "Failed to sign in with Google", true);
            console.error('Google Sign-In error:', error);
        } finally {
            setIsLoading(false);
        }
    }, [showModal, navigateTo, setIsLoading]);

    // Email/Password Sign-In handler
    const handleEmailAuth = useCallback(async (e) => {
        e.preventDefault();
        setIsLoading(true);
        const email = e.target.email.value;
        const password = e.target.password.value;

        try {
            const response = await axios.post(`${BACKEND_URL}/api/login`, { email, password });
            const data = response.data;
            setSessionId(data.session_id);
            setUserInfo(data.user);
            setIsLoggedIn(true);
            localStorage.setItem('session_id', data.session_id);
            localStorage.setItem('user_info', JSON.stringify(data.user));
            showModal("Welcome", `Signed in as ${data.user.name || data.user.email}`);
            navigateTo('upload');
        } catch (error) {
            const errorMessage = error.response?.data?.message || 'Invalid credentials. (Note: Email/Password login needs backend implementation)';
            showModal('Login Failed', errorMessage, true);
        } finally {
            setIsLoading(false);
        }
    }, [showModal, navigateTo, setIsLoading]);

    // Logout handler
    const handleLogout = useCallback(() => {
        if (window.gapi && window.gapi.auth2) {
            const auth2 = window.gapi.auth2.getAuthInstance();
            if (auth2.isSignedIn.get()) {
                auth2.signOut().then(() => {
                    console.log('User signed out from Google.');
                });
            }
        }
        setSessionId(null);
        setUserInfo(null);
        setIsLoggedIn(false);
        localStorage.removeItem('session_id');
        localStorage.removeItem('user_info');
        navigateTo('home');
        setAiAssistantOpen(false);
        setAiChatHistory([]);
        setDfInfo(null);
        setFileInfo(null);
        setCleaningRecommendations([]);
        setCleaningSteps([]);
        setGeneratedPlot(null);
        hasFetchedRecommendationsOnce.current = false; // Reset for new session
    }, [navigateTo]);

    // File upload handler
    const handleFileUpload = useCallback(async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setIsLoading(true);
        const formData = new FormData();
        formData.append('file', file);
        if (sessionId) formData.append('session_id', sessionId);

        try {
            const response = await axios.post(`${BACKEND_URL}/api/upload`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            const data = response.data;
            setSessionId(data.session_id);
            localStorage.setItem('session_id', data.session_id);
            setFileInfo({ name: file.name, rows: data.rows, columns: data.columns });
            showModal("Upload Successful", `File uploaded successfully with ${data.rows} rows and ${data.columns} columns.`);
            hasFetchedRecommendationsOnce.current = false; // Reset for new file upload
        } catch (error) {
            showModal("Upload Error", error.response?.data?.detail || "Failed to upload file", true);
            e.target.value = '';
        } finally {
            setIsLoading(false);
        }
    }, [sessionId, showModal, setIsLoading]);

    // Fetch dataset information (for Explore and Clean pages)
    const fetchDatasetInfo = useCallback(async () => {
        if (!sessionId) {
            console.warn("Attempted to fetch dataset info without a session ID.");
            return;
        }
        setIsLoading(true);
        try {
            const response = await axios.get(`${BACKEND_URL}/api/dataset-info/${sessionId}`);
            setDfInfo(response.data);
            showModal("Success", "Dataset information loaded successfully.");
        } catch (error) {
            showModal("Error", "Failed to fetch dataset information", true);
            console.error('Dataset info error:', error);
        } finally {
            setIsLoading(false);
        }
    }, [sessionId, showModal, setIsLoading]);

    // Fetch AI cleaning recommendations
    const fetchRecommendations = useCallback(async () => {
        if (!sessionId) return;
        setIsLoading(true);
        try {
            const response = await axios.get(`${BACKEND_URL}/api/recommendations/${sessionId}`);
            setCleaningRecommendations(response.data.steps || []);
            if (response.data.steps.length === 0) {
                showModal("Info", "No specific recommendations found. Your data might already be clean!");
            } else {
                showModal("Success", "AI cleaning recommendations loaded.");
            }
        } catch (error) {
            showModal("Error", `Failed to get recommendations: ${error.response?.data?.detail || error.message}`, true);
            console.error('Recommendations error:', error);
        } finally {
            setIsLoading(false);
            // Set the ref to true here after the fetch attempt completes (success or failure)
            hasFetchedRecommendationsOnce.current = true;
        }
    }, [sessionId, showModal, setIsLoading]);

    // Apply a selected cleaning step
    const applyCleaningStep = useCallback(async (step) => {
        setIsLoading(true);
        try {
            const response = await axios.post(`${BACKEND_URL}/api/apply-step/${sessionId}`, step);
            setCleaningSteps(prevSteps => [...prevSteps, response.data.step_description]);
            setCleaningRecommendations(prevRecs => prevRecs.filter(r =>
                !(r.action === step.action && r.column === step.column && r.reason === step.reason)
            ));
            await fetchDatasetInfo();
            showModal('Success', `Applied: ${response.data.step_description}`);
        } catch (error) {
            showModal("Error", `Failed to apply step: ${error.response?.data?.detail}`, true);
            console.error('Apply step error:', error);
        } finally {
            setIsLoading(false);
        }
    }, [sessionId, fetchDatasetInfo, showModal, setIsLoading]);

    // Handle AI Chat interactions
    const handleAIChat = useCallback(async (e) => {
        e.preventDefault();
        const question = aiChatInput.trim();
        if (!question) return;

        setAiChatHistory(prevHistory => [...prevHistory, { sender: 'user', message: question }]);
        setAiChatInput('');
        setAiChatHistory(prevHistory => [...prevHistory, { sender: 'ai-loading', message: '' }]);

        try {
            const response = await axios.post(`${BACKEND_URL}/api/chat/${sessionId}`, { question });
            // Clean up markdown before displaying: remove ** and ||
            const cleanedResponse = response.data.response.replace(/\*\*(.*?)\*\*/g, '$1').replace(/\|\|/g, '');
            setAiChatHistory(prevHistory => prevHistory.slice(0, -1).concat({ sender: 'ai', message: cleanedResponse }));
        } catch (error) {
            setAiChatHistory(prevHistory => prevHistory.slice(0, -1).concat({ sender: 'ai-error', message: 'Error getting AI response.' }));
            console.error('Chat error:', error);
        }
    }, [aiChatInput, sessionId, BACKEND_URL]);

    // Generate a plot based on user request
    const handleGeneratePlot = useCallback(async () => {
        const request = document.getElementById('visualization-request').value.trim();
        if (!request) {
            showModal("Input Required", "Please describe the plot you want to generate.", true);
            return;
        }

        setIsLoading(true);
        try {
            const response = await axios.post(`${BACKEND_URL}/api/generate-code/${sessionId}`, { request });
            setGeneratedPlot(response.data.plot);
            showModal("Success", "Visualization generated successfully!");
        } catch (error) {
            showModal("Error", `Failed to generate plot: ${error.response?.data?.detail}`, true);
            console.error('Plot generation error:', error);
        } finally {
            setIsLoading(false);
        }
    }, [sessionId, showModal, setIsLoading]);

    // Generate and download a PDF report
    const handleGenerateReport = useCallback(async () => {
        setIsLoading(true);
        try {
            const response = await axios.get(`${BACKEND_URL}/api/generate-report/${sessionId}`, {
                responseType: 'blob',
            });
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', 'data_report.pdf');
            document.body.appendChild(link);
            link.click();
            link.remove();
            window.URL.revokeObjectURL(url);
            showModal("Success", "PDF report downloaded successfully!");
        } catch (error) {
            showModal("Error", `Failed to generate report: ${error.response?.data?.detail}`, true);
            console.error('Report generation error:', error);
        } finally {
            setIsLoading(false);
        }
    }, [sessionId, showModal, setIsLoading]);

    // New: Handle Export Cleaned Data
    const handleExportCleanedData = useCallback(async (format) => {
        if (!sessionId) {
            showModal("Error", "No session active. Please upload a file first.", true);
            return;
        }
        setIsLoading(true);
        try {
            const response = await axios.get(`${BACKEND_URL}/api/export-data/${sessionId}?format_type=${format}`, {
                responseType: 'blob',
            });
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', `cleaned_data.${format}`);
            document.body.appendChild(link);
            link.click();
            link.remove();
            window.URL.revokeObjectURL(url);
            showModal("Success", `Cleaned data exported as ${format.toUpperCase()}!`);
        } catch (error) {
            showModal("Error", `Failed to export data: ${error.response?.data?.detail || error.message}`, true);
            console.error('Export data error:', error);
        } finally {
            setIsLoading(false);
        }
    }, [sessionId, showModal, setIsLoading]);


    // Main App component rendering logic
    const renderPage = () => {
        // Redirect to upload page if trying to access dashboard/explore/clean without a session
        if ((currentPage === 'dashboard' || currentPage === 'explore' || currentPage === 'clean') && !sessionId) {
            // Show a modal warning only once per session redirection
            if (!hasSessionWarningShown.current) {
                showModal("Session Required", "Please log in or upload a file to start a new session.", true);
                hasSessionWarningShown.current = true; // Mark warning as shown
            }
            return <UploadPage navigateTo={navigateTo} fileInfo={fileInfo} handleFileUpload={handleFileUpload} />;
        }
        
        switch (currentPage) {
            case 'home':
                return <HomePage navigateTo={navigateTo} isLoggedIn={isLoggedIn} />;
            case 'about':
                return <AboutPage navigateTo={navigateTo} />;
            case 'login':
                return <LoginPage navigateTo={navigateTo} handleGoogleSignInClick={handleGoogleSignInClick} handleEmailAuth={handleEmailAuth} />;
            case 'upload':
                return <UploadPage navigateTo={navigateTo} fileInfo={fileInfo} handleFileUpload={handleFileUpload} />;
            case 'dashboard':
                return <DashboardPage navigateTo={navigateTo} />;
            case 'explore':
                return <ExplorePage navigateTo={navigateTo} dfInfo={dfInfo} fetchDatasetInfo={fetchDatasetInfo} activeTab={activeExploreTab} setActiveTab={setActiveExploreTab} generatedPlot={generatedPlot} handleGeneratePlot={handleGeneratePlot} />;
            case 'clean':
                return <CleanPage navigateTo={navigateTo} cleaningRecommendations={cleaningRecommendations} cleaningSteps={cleaningSteps} fetchRecommendations={fetchRecommendations} applyCleaningStep={applyCleaningStep} activeTab={activeCleanTab} setActiveTab={setActiveCleanTab} handleGenerateReport={handleGenerateReport} handleExportCleanedData={handleExportCleanedData} hasFetchedRecommendationsOnce={hasFetchedRecommendationsOnce} />;
            default:
                return <HomePage navigateTo={navigateTo} isLoggedIn={isLoggedIn} />;
        }
    };

    return (
        <div className="bg-gray-900 text-white min-h-screen font-poppins">
            <Header navigateTo={navigateTo} isLoggedIn={isLoggedIn} userInfo={userInfo} handleLogout={handleLogout} />
            <main className="pt-20 pb-10">
                {renderPage()}
            </main>

            {isLoggedIn && sessionId && (currentPage === 'dashboard' || currentPage === 'explore' || currentPage === 'clean') && (
                <>
                    <button
                        id="ai-assistant-btn"
                        className="fixed bottom-6 right-6 z-30"
                        onClick={() => setAiAssistantOpen(!aiAssistantOpen)}
                    >
                        <img src={AiBotAssistantImage} alt="AI Assistant" className="w-16 h-16 cursor-pointer filter drop-shadow-lg hover:scale-110 transition-transform duration-300 animate-[pulseLight_2s_infinite]" />
                    </button>
                    <div id="ai-assistant-ui" className={`fixed bottom-24 right-6 w-80 bg-gray-800 rounded-t-xl shadow-2xl border border-pink-700 z-40 flex flex-col ${aiAssistantOpen ? 'h-96' : 'hidden'}`} style={{ maxHeight: 'calc(100vh - 10rem)' }}>
                        <div className="bg-gray-900 p-3 rounded-t-xl flex justify-between items-center border-b border-gray-700">
                            <h3 className="font-bold text-lg text-pink-400">AI Assistant</h3>
                            <button onClick={() => setAiAssistantOpen(false)} className="text-gray-400 hover:text-white transition duration-300">
                                <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
                            </button>
                        </div>
                        <div id="ai-assistant-messages" className="flex-1 overflow-y-auto p-4 space-y-4" ref={aiMessagesRef}>
                            <div className="message ai-message flex items-start space-x-2">
                                <img src={AiBotAssistantImage} alt="AI Assistant" className="ai-avatar w-8 h-8 rounded-full" />
                                <div className="message-bubble bg-gray-700 text-gray-300 p-3 rounded-lg max-w-xs">
                                    Hello! I am your AI Data Assistant. Ask me anything about your dataset, and I'll help you out!
                                </div>
                            </div>
                            {aiChatHistory.map((msg, index) => (
                                <div key={index} className={`message flex items-start space-x-2 ${msg.sender === 'user' ? 'justify-end' : ''}`}>
                                    {msg.sender !== 'user' && <img src={AiBotAssistantImage} alt="AI Assistant" className="ai-avatar w-8 h-8 rounded-full" />}
                                    <div className={`message-bubble p-3 rounded-lg max-w-xs ${msg.sender === 'user' ? 'bg-purple-600 text-white' : 'bg-gray-700 text-gray-300'}`}>
                                        {msg.sender === 'ai-loading' ? (
                                            <div className="flex space-x-2">
                                                <div className="w-2 h-2 rounded-full bg-gray-500 animate-bounce"></div>
                                                <div className="w-2 h-2 rounded-full bg-gray-500 animate-bounce animation-delay-200"></div>
                                                <div className="w-2 h-2 rounded-full bg-gray-500 animate-bounce animation-delay-400"></div>
                                            </div>
                                        ) : (
                                            msg.message
                                        )}
                                    </div>
                                    {msg.sender === 'user' && <div className="ai-avatar w-8 h-8 rounded-full" />}
                                </div>
                            ))}
                        </div>
                        <form onSubmit={handleAIChat} className="ai-assistant-input p-3 border-t border-gray-700 flex">
                            <input
                                type="text"
                                value={aiChatInput}
                                onChange={(e) => setAiChatInput(e.target.value)}
                                placeholder="Ask a question about your data..."
                                className="flex-grow p-3 rounded-lg bg-gray-700 text-white placeholder-gray-400 border border-gray-600 focus:ring-pink-500 focus:border-pink-500"
                            />
                            <button type="submit" className="ml-2 py-3 px-4 bg-pink-600 text-white rounded-lg hover:bg-pink-700 transition duration-300">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" /></svg>
                            </button>
                        </form>
                    </div>
                </>
            )}
            <Loader isLoading={isLoading} />
            <Modal modal={modal} closeModal={closeModal} />
        </div>
    );
}

// Helper function to format bytes for display
const formatBytes = (bytes, decimals = 2) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
};

// --- React Components ---

const Header = ({ navigateTo, isLoggedIn, userInfo, handleLogout }) => {
    return (
        <header id="header-nav" className="fixed w-full top-0 z-50 bg-gray-950/70 backdrop-blur-md shadow-lg p-4 flex justify-between items-center transition-all duration-300">
            <h1 className="text-3xl sm:text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-purple-500">DataPreProAI</h1>
            <nav className="flex items-center space-x-2 sm:space-x-4">
                <button onClick={() => navigateTo('home')} className="nav-item">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /><polyline points="9 22 9 12 15 12 15 22" /></svg>
                    <span className="hidden sm:inline">Home</span>
                </button>
                <button onClick={() => navigateTo('about')} className="nav-item">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><path d="M12 16v-4" /><path d="M12 8h.01" /></svg>
                    <span className="hidden sm:inline">About Us</span>
                </button>
                {isLoggedIn ? (
                    <button onClick={handleLogout} className="nav-item">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" /><polyline points="16 17 21 12 16 7" /><line x1="21" y1="12" x2="9" y2="12" /></svg>
                        <span className="hidden sm:inline">Logout</span>
                    </button>
                ) : (
                    <button onClick={() => navigateTo('login')} className="nav-item">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4" /><polyline points="10 17 15 12 10 7" /><line x1="15" y1="12" x2="3" y2="12" /></svg>
                        <span className="hidden sm:inline">Login</span>
                    </button>
                )}
            </nav>
        </header>
    );
};

const Loader = ({ isLoading }) => {
    return isLoading && (
        <div id="loader" className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-gray-800 p-8 rounded-xl shadow-2xl border border-pink-700 flex flex-col items-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="animate-spin h-16 w-16 text-pink-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 1 1-6.219-8.56" /></svg>
                <p className="mt-4 text-gray-300">Processing your request...</p>
            </div>
        </div>
    );
};

const Modal = ({ modal, closeModal }) => {
    return modal.isVisible && (
        <div id="modal-container" className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-xl shadow-2xl p-6 max-w-md w-full relative border border-pink-700">
                <h3 id="modal-title" className={`text-2xl font-bold mb-4 ${modal.isError ? 'text-red-400' : 'text-pink-400'}`}>{modal.title}</h3>
                <div id="modal-body" className={`mb-6 ${modal.isError ? 'text-red-400' : 'text-gray-300'}`}>
                    <p>{modal.message}</p>
                </div>
                <button onClick={closeModal} className="absolute top-4 right-4 text-gray-400 hover:text-white transition duration-300">
                    <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
                </button>
                <button onClick={closeModal} className="w-full py-2 px-4 bg-pink-600 text-white rounded-lg hover:bg-pink-700 transition duration-300">OK</button>
            </div>
        </div>
    );
};

const HomePage = ({ navigateTo, isLoggedIn }) => (
    <section id="landing-page" className="page active animate-fade-in">
        <div className="min-h-screen flex flex-col items-center justify-center text-white p-4 text-center">
            <h1 className="text-5xl md:text-7xl font-extrabold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-purple-500 animate-fade-in-up">
                DataPreProAI
            </h1>
            <p className="text-xl md:text-2xl text-gray-300 mb-12 max-w-2xl animate-fade-in-up animation-delay-200">
                Your AI-powered assistant for effortless data preprocessing and preparation. Here you can prepare your data with the assistance of AI.
            </p>
            <button
                id="start-cleaning-btn"
                onClick={() => navigateTo(isLoggedIn ? 'upload' : 'login')}
                className="px-8 py-4 sm:px-12 sm:py-5 bg-gradient-to-r from-pink-600 to-purple-500 text-white text-xl sm:text-3xl font-bold rounded-xl shadow-xl hover:scale-105 transition-all duration-300 transform-gpu animate-bounce-in"
            >
                Start Cleaning
            </button>
        </div>
    </section>
);

const AboutPage = ({ navigateTo }) => (
    <section id="about-page" className="page animate-fade-in-up">
        <div className="container mx-auto px-4 py-8">
            <button onClick={() => navigateTo('home')} className="back-btn mb-8 flex items-center space-x-2 text-gray-400 hover:text-white">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m12 19-7-7 7-7" /><path d="M19 12H5" /></svg>
                <span>Back to Home</span>
            </button>
            <div className="max-w-4xl mx-auto">
                <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-purple-500 mb-8 text-center">
                    About DataPreproAI AI
                </h1>
                <div className="bg-gradient-to-br from-gray-800 to-purple-900 rounded-xl shadow-2xl p-8 mb-12 border border-pink-700">
                    <h2 className="text-3xl font-bold text-pink-400 mb-6">Our Mission</h2>
                    <p className="text-gray-300 text-lg leading-relaxed mb-6">
                        At DataPreproAI AI, we believe that data analysis should be accessible to everyone, regardless of technical expertise. Our mission is to democratize data science by providing intuitive, AI-powered tools that make data cleaning, exploration, and visualization effortless.
                    </p>
                    <p className="text-gray-300 text-lg leading-relaxed">
                        Whether you're a seasoned data scientist or just starting your data journey, DataPreProAI empowers you to extract meaningful insights from your data with minimal effort.
                    </p>
                </div>
            </div>
        </div>
    </section>
);

const LoginPage = ({ navigateTo, handleGoogleSignInClick, handleEmailAuth }) => (
    <section id="login-page" className="page animate-fade-in-up">
        <div className="min-h-screen flex flex-col items-center justify-center p-4">
            <button onClick={() => navigateTo('home')} className="back-btn absolute top-24 left-4 flex items-center space-x-2 text-gray-400 hover:text-white">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m12 19-7-7 7-7" /><path d="M19 12H5" /></svg>
                <span>Back to Home</span>
            </button>
            <div className="bg-gradient-to-br from-gray-800 to-purple-900 p-8 rounded-2xl shadow-2xl text-center max-w-md w-full border border-pink-700 animate-scale-in">
                <h2 className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-purple-500 mb-4">
                    Welcome to DataPreproAI
                </h2>
                <p className="text-gray-400 mb-8 text-lg">
                    Sign in or register to start analyzing your data
                </p>
                <button
                    onClick={handleGoogleSignInClick}
                    className="w-full py-3 px-6 bg-pink-600 text-white rounded-lg shadow-lg hover:bg-pink-700 transition duration-300 flex items-center justify-center space-x-3 text-lg font-semibold mb-4"
                >
                    <img src={GoogleSignInIcon} alt="Google" className="w-6 h-6" />
                    <span>Sign in with Google</span>
                </button>

                <div className="flex items-center my-6">
                    <hr className="flex-grow border-gray-600" />
                    <span className="mx-4 text-gray-400">or continue with email</span>
                    <hr className="flex-grow border-gray-600" />
                </div>
                <form id="email-auth-form" onSubmit={handleEmailAuth}>
                    <input required type="email" name="email" placeholder="Email" className="w-full p-3 mb-4 rounded-lg bg-gray-700 text-white placeholder-gray-400 border border-gray-600 focus:ring-pink-500 focus:border-pink-500" />
                    <input required type="password" name="password" placeholder="Password" className="w-full p-3 mb-6 rounded-lg bg-gray-700 text-white placeholder-gray-400 border border-gray-600 focus:ring-pink-500 focus:border-pink-500" />
                    <button type="submit" className="w-full py-3 px-6 bg-purple-600 text-white rounded-lg shadow-lg hover:bg-purple-700 transition duration-300 text-lg font-semibold">
                        Sign in / Register
                    </button>
                </form>
            </div>
        </div>
    </section>
);

const UploadPage = ({ navigateTo, fileInfo, handleFileUpload }) => { // Removed showMissingSessionWarning prop
    const fileInputRef = useRef(null);

    const handleDragOver = (e) => {
        e.preventDefault();
        e.currentTarget.classList.add('border-pink-500', 'text-pink-400');
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        e.currentTarget.classList.remove('border-pink-500', 'text-pink-400');
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.currentTarget.classList.remove('border-pink-500', 'text-pink-400');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload({ target: { files: files } });
        }
    };

    return (
        <section id="upload-page" className="page animate-fade-in">
            <div className="container mx-auto px-4 py-8">
                <button onClick={() => navigateTo('home')} className="back-btn mb-8 flex items-center space-x-2 text-gray-400 hover:text-white">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m12 19-7-7 7-7" /><path d="M19 12H5" /></svg>
                    <span>Back to Home</span>
                </button>
                <div className="bg-gradient-to-br from-gray-800 to-purple-900 rounded-2xl shadow-2xl p-6 md:p-10 border border-pink-700 animate-scale-in max-w-4xl mx-auto">
                    <h2 className="text-3xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-purple-500">Upload Your Data</h2>
                    <p className="text-gray-400 mb-8">Select a CSV or Excel file to get started with AI-powered data preparation.</p>
                    <div
                        id="file-upload-area"
                        className={`flex flex-col items-center justify-center p-8 border-2 border-dashed border-gray-600 rounded-xl text-gray-400 h-64 bg-gray-800 transition-all duration-300 ${fileInfo ? 'hidden' : ''}`}
                        onClick={() => fileInputRef.current?.click()}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mb-4 text-pink-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" x2="12" y1="3" y2="15" /></svg>
                        <p className="text-xl font-semibold mb-2">Drag & Drop or <span className="text-pink-400">Choose File</span></p>
                        <p className="text-sm">CSV or Excel up to 50MB</p>
                        <input type="file" ref={fileInputRef} className="hidden" accept=".csv, .xlsx, .xls" onChange={handleFileUpload} />
                    </div>
                    {fileInfo && (
                        <div id="file-uploaded-info" className="mt-6">
                            <div className="bg-gray-800 p-6 rounded-lg shadow-md mb-6 border border-pink-700 animate-fade-in-up">
                                <h3 className="text-xl font-semibold text-pink-400 mb-2">File Uploaded: {fileInfo.name}</h3>
                                <div className="grid grid-cols-2 gap-2 text-gray-300">
                                    <p>Rows: <span className="font-bold">{fileInfo.rows}</span></p>
                                    <p>Columns: <span className="font-bold">{fileInfo.columns}</span></p>
                                </div>
                            </div>
                            <button
                                onClick={() => navigateTo('dashboard')}
                                className="w-full py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white font-bold rounded-lg shadow-xl hover:scale-105 transition duration-300 transform-gpu animate-bounce-in animation-delay-400"
                            >
                                Prepare the Data
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </section>
    );
};

const DashboardPage = ({ navigateTo }) => (
    <section id="dashboard-page" className="page animate-fade-in-up">
        <div className="container mx-auto px-4 py-8">
            <div className="flex justify-between items-center mb-8">
                <button onClick={() => navigateTo('upload')} className="back-btn flex items-center space-x-2 text-gray-400 hover:text-white">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m12 19-7-7 7-7" /><path d="M19 12H5" /></svg>
                    <span>Back to Upload</span>
                </button>
                <h2 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-purple-500">Dataset Dashboard</h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="flex flex-col bg-gradient-to-br from-gray-800 to-purple-900 p-8 rounded-2xl shadow-2xl border border-pink-700">
                    <h3 className="text-2xl font-bold text-pink-400 mb-4">Clean Data</h3>
                    <p className="text-gray-300 mb-6">Use AI recommendations to automatically clean and prepare your dataset for analysis.</p>
                    <button onClick={() => navigateTo('clean')} className="mt-auto w-full py-3 px-6 bg-purple-600 text-white rounded-lg shadow-lg hover:bg-purple-700 transition duration-300 text-lg font-semibold">
                        Clean Data
                    </button>
                </div>
                <div className="flex flex-col bg-gradient-to-br from-gray-800 to-pink-900 p-8 rounded-2xl shadow-2xl border border-pink-700">
                    <h3 className="text-2xl font-bold text-pink-400 mb-4">Explore Data</h3>
                    <p className="text-gray-300 mb-6">Dive into your data with interactive previews, summary statistics, and AI-generated visualizations.</p>
                    <button onClick={() => navigateTo('explore')} className="mt-auto w-full py-3 px-6 bg-pink-600 text-white rounded-lg shadow-lg hover:bg-pink-700 transition duration-300 text-lg font-semibold">
                        Explore Data
                    </button>
                </div>
            </div>
        </div>
    </section>
);

const ExplorePage = ({ navigateTo, dfInfo, fetchDatasetInfo, activeTab, setActiveTab, generatedPlot, handleGeneratePlot }) => {
    useEffect(() => {
        if (!dfInfo) {
            fetchDatasetInfo();
        }
    }, [fetchDatasetInfo, dfInfo]);

    const renderContent = () => {
        if (!dfInfo) return <p className="text-center text-gray-400">Loading data info...</p>;
        switch (activeTab) {
            case 'preview':
                const data = dfInfo.df_head_json ? JSON.parse(dfInfo.df_head_json) : { data: [], columns: [] };
                const first10Rows = data.data.slice(0, 10);
                const columns = data.columns;
                return (
                    <>
                        <div className="overflow-x-auto rounded-lg shadow-md border border-gray-700">
                            <table className="min-w-full divide-y divide-gray-700">
                                <thead className="bg-gray-700">
                                    <tr>
                                        {columns.map(col => <th key={col} className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">{col}</th>)}
                                    </tr>
                                </thead>
                                <tbody className="bg-gray-800 divide-y divide-gray-700">
                                    {first10Rows.map((row, rowIndex) => (
                                        <tr key={rowIndex}>
                                            {row.map((cell, cellIndex) => <td key={cellIndex} className="px-4 py-3 whitespace-nowrap text-sm text-gray-200">{cell !== null ? cell : <span className="text-gray-500">null</span>}</td>)}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                        <p className="mt-2 text-sm text-gray-400">Showing first 10 rows of {dfInfo.rows} total rows</p>
                    </>
                );
            case 'summary':
                return (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="bg-gray-700 p-4 rounded-lg">
                            <h4 className="text-lg font-semibold text-purple-300 mb-2">Basic Info</h4>
                            <ul className="space-y-2 text-gray-300">
                                <li>Rows: <span className="font-bold">{dfInfo.rows}</span></li>
                                <li>Columns: <span className="font-bold">{dfInfo.columns}</span></li>
                                <li>Missing Values: <span className="font-bold">{dfInfo.missing_values}</span></li>
                                <li>Memory Usage: <span className="font-bold">{formatBytes(dfInfo.memory_usage * 1024 * 1024)}</span></li>
                            </ul>
                        </div>
                        <div className="bg-gray-700 p-4 rounded-lg">
                            <h4 className="text-lg font-semibold text-purple-300 mb-2">Column Types</h4>
                            <ul className="space-y-2 text-gray-300">
                                {Object.entries(dfInfo.dtypes).map(([col, type]) => <li key={col}>{col}: <span className="text-pink-300">{type}</span></li>)}
                            </ul>
                        </div>
                        <div className="bg-gray-700 p-4 rounded-lg md:col-span-2">
                            <h4 className="text-lg font-semibold text-purple-300 mb-2">Missing Values per Column</h4>
                            <div className="overflow-x-auto">
                                <table className="min-w-full">
                                    <thead>
                                        <tr>
                                            <th className="px-4 py-2 text-left text-gray-300">Column</th>
                                            <th className="px-4 py-2 text-left text-gray-300">Missing Count</th>
                                            <th className="px-4 py-2 text-left text-gray-300">Percentage</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {Object.entries(dfInfo.missing_values_per_column || {}).map(([col, count]) => (
                                            <tr key={col}>
                                                <td className="px-4 py-2 text-gray-300">{col}</td>
                                                <td className="px-4 py-2 text-gray-300">{count}</td>
                                                <td className="px-4 py-2 text-gray-300">{((count / dfInfo.rows) * 100).toFixed(2)}%</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                );
            case 'visualize':
                return (
                    <div className="space-y-6">
                        <div className="bg-gray-800 p-6 rounded-lg shadow-md border border-gray-700">
                            <h3 className="text-xl font-bold text-pink-400 mb-4">Generate Visualization</h3>
                            <div className="flex flex-col md:flex-row gap-4">
                                <input id="visualization-request" type="text" placeholder="Describe the plot you want (e.g., 'Show a bar chart of sales by region')" className="flex-grow p-3 rounded-lg bg-gray-700 text-white placeholder-gray-400 border border-gray-600 focus:ring-pink-500 focus:border-pink-500" />
                                <button onClick={handleGeneratePlot} id="generate-plot-btn" className="px-6 py-3 bg-pink-600 text-white rounded-lg shadow-md hover:bg-pink-700 transition duration-300 font-semibold whitespace-nowrap">
                                    Generate Plot
                                </button>
                            </div>
                        </div>
                        <div id="visualization-output">
                            {generatedPlot ? (
                                <div className="bg-gray-800 p-6 rounded-lg shadow-md border border-gray-700">
                                    <h3 className="text-xl font-bold text-pink-400 mb-4">Visualization</h3>
                                    <div className="bg-white p-4 rounded-lg">
                                        <img src={`data:image/png;base64,${generatedPlot}`} alt="Generated visualization" className="w-full" />
                                    </div>
                                </div>
                            ) : (
                                <div className="bg-gray-800 p-8 rounded-lg shadow-md border border-gray-700 text-center">
                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mx-auto text-gray-500 mb-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>
                                    <p className="text-gray-400">No visualization generated yet. Enter a description above to create one.</p>
                                </div>
                            )}
                        </div>
                    </div>
                );
            default:
                return null;
        }
    };

    return (
        <section id="explore-page" className="page animate-fade-in-up">
            <div className="container mx-auto px-4 py-8">
                <button onClick={() => navigateTo('dashboard')} className="back-btn mb-8 flex items-center space-x-2 text-gray-400 hover:text-white">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m12 19-7-7 7-7" /><path d="M19 12H5" /></svg>
                    <span>Back to Dashboard</span>
                </button>
                <h2 className="text-3xl font-bold mb-8 text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-purple-500">Explore Your Data</h2>
                <div className="bg-gradient-to-br from-gray-800 to-purple-900 p-8 rounded-2xl shadow-2xl border border-pink-700">
                    <div id="explore-tabs" className="flex border-b border-gray-700 mb-6">
                        <button onClick={() => setActiveTab('preview')} className={`explore-tab ${activeTab === 'preview' ? 'active-tab' : ''} px-4 py-2 text-sm font-medium rounded-t-lg`}>Data Preview</button>
                        <button onClick={() => setActiveTab('summary')} className={`explore-tab ${activeTab === 'summary' ? 'active-tab' : ''} px-4 py-2 text-sm font-medium rounded-t-lg`}>Summary Stats</button>
                        <button onClick={() => setActiveTab('visualize')} className={`explore-tab ${activeTab === 'visualize' ? 'active-tab' : ''} px-4 py-2 text-sm font-medium rounded-t-lg`}>Visualize</button>
                    </div>
                    <div id="explore-content-container">
                        {renderContent()}
                    </div>
                </div>
            </div>
        </section>
    );
};

const CleanPage = ({ navigateTo, cleaningRecommendations, cleaningSteps, fetchRecommendations, applyCleaningStep, activeTab, setActiveTab, handleGenerateReport, handleExportCleanedData, hasFetchedRecommendationsOnce }) => {
    useEffect(() => {
        // Only fetch recommendations if the 'recommendations' tab is active AND
        // if recommendations haven't been loaded yet AND we haven't tried fetching them before in this session
        // Added a check for hasFetchedRecommendationsOnce.current to ensure it's accessed safely.
        if (activeTab === 'recommendations' && cleaningRecommendations.length === 0 && hasFetchedRecommendationsOnce && !hasFetchedRecommendationsOnce.current) {
            fetchRecommendations();
            // The ref update should happen inside the fetchRecommendations function's success/finally block
            // to ensure it's marked as fetched only after the API call completes (or attempts to).
        }
    }, [activeTab, fetchRecommendations, cleaningRecommendations.length, hasFetchedRecommendationsOnce]); // Added hasFetchedRecommendationsOnce to dependencies

    const renderContent = () => {
        switch (activeTab) {
            case 'recommendations':
                return (
                    <div id="recommendations-container" className="space-y-4">
                        <div className="flex justify-between items-center">
                            <h3 className="text-xl font-bold text-pink-400">AI-Powered Cleaning Recommendations</h3>
                            <button onClick={fetchRecommendations} id="refresh-recommendations-btn" className="px-4 py-2 bg-purple-600 text-white rounded-lg shadow-md hover:bg-purple-700 transition duration-300 text-sm font-semibold">
                                Refresh Recommendations
                            </button>
                        </div>
                        <div id="recommendations-list" className="space-y-3">
                            {cleaningRecommendations.length > 0 ?
                                cleaningRecommendations.map((step, index) => (
                                    <div key={index} className="bg-gray-800 p-4 rounded-lg shadow-md border border-pink-700 flex justify-between items-center animate-fade-in-up">
                                        <div className="flex items-center space-x-3">
                                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-yellow-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 6c0 1.9.8 3.7 2.2 5l-2.4 2.4" /><path d="M4.2 11.2a9 9 0 0 0 14.6 6.3l.7.7.7.7" /><path d="M14.5 18a.5.5 0 1 0 0-1 .5.5 0 0 0 0 1z" /><path d="M12 21a2 2 0 0 0-2 2v1" /><path d="M16 21a2 2 0 0 0-2-2v-1" /><path d="M19 12h-2" /><path d="M5 12H3" /></svg>
                                            <div>
                                                <p className="text-gray-300"><span className="font-bold">{step.action}</span> on column <span className="text-pink-400 font-mono">{step.column}</span></p>
                                                <p className="text-sm text-gray-400">{step.reason}</p>
                                            </div>
                                        </div>
                                        <button onClick={() => applyCleaningStep(step)} data-index={index} className="apply-step-btn px-3 py-1 bg-pink-600 text-white rounded-lg text-sm hover:bg-pink-700 transition duration-300">
                                            Apply
                                        </button>
                                    </div>
                                )) :
                                <div className="text-center p-8 text-gray-400">No recommendations found. Click "Refresh Recommendations" to generate some.</div>
                            }
                        </div>
                        <div id="applied-steps-list" className="mt-8 space-y-3">
                            <h4 className="text-xl font-bold text-pink-400">Applied Steps</h4>
                            {cleaningSteps.length > 0 ?
                                cleaningSteps.map((step, index) => (
                                    <div key={index} className="bg-gray-800 p-4 rounded-lg shadow-md border border-green-700 flex items-center animate-fade-in-up">
                                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-3 text-green-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
                                        <span className="text-gray-300">{step}</span>
                                    </div>
                                )) :
                                <div className="text-center p-4 text-gray-400">No cleaning steps applied yet.</div>
                            }
                        </div>
                        {/* New Export Cleaned Data Buttons */}
                        <div className="mt-8 flex flex-col md:flex-row gap-4 justify-center">
                            <button
                                onClick={() => handleExportCleanedData('csv')}
                                className="px-6 py-3 bg-blue-600 text-white rounded-lg shadow-md hover:bg-blue-700 transition duration-300 font-semibold"
                            >
                                Export as CSV
                            </button>
                        </div>
                    </div>
                );
            case 'report':
                return (
                    <div className="bg-gray-800 p-8 rounded-lg shadow-md border border-gray-700 text-center animate-fade-in-up">
                        <h3 className="text-xl font-bold text-pink-400 mb-4">Generate Comprehensive Report</h3>
                        <p className="text-gray-300 mb-6">Create a detailed PDF report of your dataset, including:</p>
                        <ul className="text-left text-gray-400 mb-6 space-y-2 max-w-md mx-auto">
                            <li className="flex items-center"><svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-green-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>Dataset overview and statistics</li>
                            <li className="flex items-center"><svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-green-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>Data quality assessment</li>
                            <li className="flex items-center"><svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-green-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>Cleaning steps applied</li>
                            <li className="flex items-center"><svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-green-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>AI-generated insights</li>
                        </ul>
                        <button onClick={handleGenerateReport} id="generate-report-btn" className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg shadow-md hover:from-purple-700 hover:to-pink-700 transition duration-300 font-semibold">
                            Generate PDF Report
                        </button>
                    </div>
                );
            default:
                return null;
        }
    };

    return (
        <section id="clean-page" className="page animate-fade-in-up">
            <div className="container mx-auto px-4 py-8">
                <button onClick={() => navigateTo('dashboard')} className="back-btn mb-8 flex items-center space-x-2 text-gray-400 hover:text-white">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m12 19-7-7 7-7" /><path d="M19 12H5" /></svg>
                    <span>Back to Dashboard</span>
                </button>
                <h2 className="text-3xl font-bold mb-8 text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-purple-500">Clean Your Data</h2>
                <div className="bg-gradient-to-br from-gray-800 to-purple-900 p-8 rounded-2xl shadow-2xl border border-pink-700">
                    <div id="clean-tabs" className="flex border-b border-gray-700 mb-6">
                        <button onClick={() => setActiveTab('recommendations')} className={`clean-tab ${activeTab === 'recommendations' ? 'active-tab' : ''} px-4 py-2 text-sm font-medium rounded-t-lg`}>AI Recommendations for Cleaning</button>
                        <button onClick={() => setActiveTab('report')} className={`clean-tab ${activeTab === 'report' ? 'active-tab' : ''} px-4 py-2 text-sm font-medium rounded-t-lg`}>Generate Report</button>
                    </div>
                    <div id="clean-content-container">
                        {renderContent()}
                    </div>
                </div>
            </div>
        </section>
    );
};

export default App;
