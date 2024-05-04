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
    <form onSubmit={handleSubmit}>
      <label>
        Type of garment:
        <select value={garmentType} onChange={e => setGarmentType(e.target.value)}>
          <option value="">--Select a garment type--</option>
          {Object.entries(garmentTypes).map(([key, value]) => (
            <option key={key} value={value}>{key}</option>
          ))}
        </select>
      </label>
      <label>
        Season:
       <select value={seasons} onChange={e => setSeasons(e.target.value)}>
          <option value="">--Select a season--</option>
          {Object.entries(seasons).map(([key, value]) => (
            <option key={key} value={value}>{key}</option>
          ))}
        </select>
      </label>
      <label>
        Color:
       <select value={colors} onChange={e => setColors(e.target.value)}>
          <option value="">--Select a color--</option>
          {Object.entries(colors).map(([key, value]) => (
            <option key={key} value={value}>{key}</option>
          ))}
        </select>
      </label>
      <input type="submit" value="Upload" />
    </form>
  );
};

export default UploadClothesForm;