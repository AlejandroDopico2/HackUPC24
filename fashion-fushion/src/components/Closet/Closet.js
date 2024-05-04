import React from 'react';
import { useNavigate } from 'react-router-dom';

const Closet = () => {
  const navigate = useNavigate();

  const handleUploadClick = () => {
    navigate('/uploadClothesForm');
  };

  return (
    <div>
      <div>Closet Component</div>
     <div className="button-container">
  <button className="upload-button" onClick={handleUploadClick}>Upload Clothes</button>
</div>
    </div>
  );
};

export default Closet;