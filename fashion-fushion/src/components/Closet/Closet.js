import React from 'react';
import { useNavigate } from 'react-router-dom';

const Closet = () => {
  const navigate = useNavigate();

  const handleUploadClick = () => {
    navigate('/uploadClothesForm');
  };

  return (
    <div className="form-container">
      <div className="form-form">
        <h1 className="title">Closet Component</h1>
        <div className="button-container">
          <button className="upload-button" onClick={handleUploadClick}>Upload Clothes</button>
        </div>
      </div>
    </div>
  );
}

export default Closet;