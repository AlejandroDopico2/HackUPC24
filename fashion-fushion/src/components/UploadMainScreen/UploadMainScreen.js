import React, { useState, useEffect } from 'react';
import { supabase } from '../../utils/supabase';
import Login from '../Login/Login';

function UploadMainScreen() {
  const [image, setImage] = useState('');
  const [originalImage, setOriginalImage] = useState('');
  let imagen = null;
  const [session, setSession] = useState(null);

  useEffect(() => {
    supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
    });
  }, []);

  function dataURLtoFile(dataurl, filename) {

    var arr = dataurl.split(','),
      mime = arr[0].match(/:(.*?);/)[1],
      bstr = atob(arr[1]),
      n = bstr.length,
      u8arr = new Uint8Array(n);
  
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
  
    return new File([u8arr], filename, {
      type: mime
    });
  }

  const handleFileUpload = async (event) => {
    event.preventDefault();

    if (event.target.files.length === 0) {
      console.error('No file selected');
      return;
    }

    const formData = new FormData();
    formData.append('file', event.target.files[0]);
    setOriginalImage(URL.createObjectURL(event.target.files[0]));
    
    try {
      const response = await fetch('http://127.0.0.1:8000/getRelatedGarments', {
        method: 'POST',
        body: formData
      });
    
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
    
      const data = await response.json();
      console.log(data);
      setImage(data.image);
      console.log("IMAGENNNN")
      imagen = data.image;
      console.log(imagen)

    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  return (
    <div className="App">
      <header>
        {!session ? (
          <Login />
        ) : (
          <div className="form-container">
            <form>
              <input type="file" onChange={handleFileUpload} />
            </form>
            <div className="image-divided">
              {originalImage && <img src={originalImage} width={200}/>}
              {image && <img src={`data:image/png;base64, ${image}`} width={200}/>}
            </div>
          </div>
        )}
      </header>
      <div>
      </div>
    </div>
  );
}

export default UploadMainScreen;