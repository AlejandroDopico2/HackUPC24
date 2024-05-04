import React from 'react';
import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { supabase } from './utils/supabase';
import './App.css';
import Closet from "./components/Closet/Closet";
import Favourites from "./components/Favourites/Favourites";
import SideMenu from "./components/SideMenu/SideMenu";
import UploadClothesForm from "./components/UploadClothesForm/UploadClothesForm";
import SignUp from './components/SignUp/SignUp';
import Login from './components/Login/Login';
import UploadMainScreen from './components/UploadMainScreen/UploadMainScreen';

function App() {
  const [session, setSession] = useState(null);

  useEffect(() => {
    supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
    });
  }, []);

  return (
    <BrowserRouter>
      <SideMenu />
      <Routes>
        <Route path="/" element={session ? <UploadMainScreen /> : <Login />} />
        <Route path="/" element={<Login />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<SignUp />} />
        <Route path="/upload" element={session ? <UploadMainScreen /> : <Login />} />
        <Route path="/closet" element={<Closet />} />
        <Route path="/favourites" element={<Favourites />} />
        <Route path="/uploadClothesForm" element={<UploadClothesForm />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;