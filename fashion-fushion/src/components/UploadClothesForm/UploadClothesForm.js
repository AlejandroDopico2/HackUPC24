import React, { useEffect, useState } from 'react';
import { supabase } from "../../utils/supabase";
import { useNavigate } from 'react-router-dom';

const UploadClothesForm = () => {
  const navigate = useNavigate();
  const [errorMessage, setErrorMessage] = useState('');
  const [garmentType, setGarmentType] = useState('');
  const [season, setSeason] = useState('');
  const [seasons, setSeasons] = useState('');
  const [colors, setColors] = useState('');
  const [color, setColor] = useState('');
  const [garmentTypes, setGarmentTypes] = useState([]);
  const [image, setImage] = useState(null);
  const [user, setUser] = useState(null);

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

      supabase.auth.getUser().then((data) => {
        setUser(data.data.user);
        console.log(data.data.user)
      });
  }, []);

  const handleImageChange = async (e) => {
   setImage(e.target.files[0]);
   console.log(image);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    console.log(`Garment Type: ${garmentType}, Season: ${season}, Color: ${color}`);
    console.log(`Image: `, image);

    supabase.storage.from('images2').upload( image.name, image).then(({data, error}) => {
      if (error) {
          setErrorMessage(error.message);
          console.error('There was an error uploading the image:', error.message);
          return;
      }

      supabase
          .from('Garment')
          .insert([
              {
                  type: garmentType.valueOf(),
                  season: season.valueOf(),
                  color: color.valueOf(),
                  image: data.fullPath,
                  user_id: user.id
              },
          ])
          .then(({data: insertData, error: insertError}) => {
              if (insertError) {
                  setErrorMessage(error.message);
                  console.error('There was an error saving the garment:', insertError.message);
              } else {
                  console.log('Garment saved successfully:', insertData);
              }
          });
      });

      if (errorMessage != '') {
        navigate('/closet');
      }
  };

   return (
    <div>
      <div className="header">
        <h1>Upload Garment</h1>
      </div>
      <form onSubmit={handleSubmit} className="upload-form">
        <label className="upload-label">
          Type of garment
          <select value={garmentType} onChange={e => setGarmentType(e.target.value)}>
            <option value="">Select a garment type</option>
            {Object.entries(garmentTypes).map(([key, value]) => (
              <option key={key} value={value}>{key}</option>
            ))}
          </select>
        </label>
        <label className="upload-label">
          Season
        <select value={season} onChange={e => setSeason(e.target.value)}>
            <option value="">Select a season</option>
            {Object.entries(seasons).map(([key, value]) => (
              <option key={key} value={value}>{key}</option>
            ))}
          </select>
        </label>
        <label className="upload-label">
          Color
        <select value={color} onChange={e => setColor(e.target.value)}>
            <option value="">Select a color</option>
            {Object.entries(colors).map(([key, value]) => (
              <option key={key} value={value}>{key}</option>
            ))}
          </select>
        </label>
        {errorMessage && <div className="error-message">{errorMessage}</div>}
        <div className="upload-image">
          <input type="file" onChange={handleImageChange} />
          <button type="submit">Upload</button>
        </div>
      </form>
    </div>
  );
};

export default UploadClothesForm;