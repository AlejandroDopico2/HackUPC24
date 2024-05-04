import React, {useEffect, useState} from 'react';


const UploadClothesForm = () => {
  const [garmentType, setGarmentType] = useState('');
  const [seasons, setSeasons] = useState('');
  const [colors, setColors] = useState('');
  const [garmentTypes, setGarmentTypes] = useState([]);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/getGarmentTypes')
      .then(response => response.json())
      .then(data => setGarmentTypes(data))
      .catch(error => console.error('There was an error!', error));


    fetch('http://127.0.0.1:8000/getSeasons')
      .then(response => response.json())
      .then(data => setSeasons(data))
      .catch(error => console.error('There was an error!', error));

      fetch('http://127.0.0.1:8000/getColors')
      .then(response => response.json())
      .then(data => setColors(data))
      .catch(error => console.error('There was an error!', error));
  }, []);

  const handleSubmit = (event) => {
    event.preventDefault();
    console.log(`Garment Type: ${garmentType}, Season: ${seasons}, Color: ${colors}`);
  };

  return (
    <div className="form-container">
      <form className="form-form">
        <h2>Upload Clothes</h2>
        <div className="form-group">
          <input type="text" id="clothesName" name="clothesName" required />
          <label htmlFor="clothesName">Clothes Name</label>
        </div>
        <div className="form-group">
          <input type="text" id="clothesType" name="clothesType" required />
          <label htmlFor="clothesType">Clothes Type</label>
        </div>
        <div className="form-group">
          <input type="text" id="clothesSize" name="clothesSize" required />
          <label htmlFor="clothesSize">Clothes Size</label>
        </div>
        <div className="form-group">
          <input type="text" id="clothesColor" name="clothesColor" required />
          <label htmlFor="clothesColor">Clothes Color</label>
        </div>
        <button type="submit">Upload</button>
      </form>
    </div>
  );
};

export default UploadClothesForm;