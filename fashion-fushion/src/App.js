import logo from './logo.svg';
import './App.css';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Closet from "./components/Closet/Closet";
import Favourites from "./components/Favourites/Favourites";
import SideMenu from "./components/SideMenu/SideMenu";
import FileInput from "./components/FileInput/FileInput";
import UploadClothesForm from "./components/UploadClothesForm/UploadClothesForm";

function App() {

  return (
    <Router>
      <div className="App">
          <SideMenu />
          <Routes>
            <Route path="/closet" element={<Closet />} />
            <Route path="/favourites" element={<Favourites />} />
            <Route path="/upload" element={<FileInput />} />
              <Route path="/uploadClothesForm" element={<UploadClothesForm />} />

          </Routes>
      </div>
    </Router>
  );
}

export default App;
