import React from 'react';
import { useNavigate } from 'react-router-dom';

const Closet = () => {
  const navigate = useNavigate();

  const handleUploadClick = () => {
    navigate('/uploadClothesForm');
  };

  return (
    <div className="header">
      <h1>Closet</h1>
      <div className="button-container">
        <button onClick={handleUploadClick}>Upload Garment</button>
      </div>
    </div>
  );
}

export default Closet;