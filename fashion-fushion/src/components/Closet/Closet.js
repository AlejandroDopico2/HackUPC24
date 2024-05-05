import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { supabase } from "../../utils/supabase";

const Closet = () => {
  const navigate = useNavigate();
  const [garments, setGarments] = useState([]);

 useEffect(() => {
  supabase.auth.getUser().then(async (data) => {
    try {
      const { data: garmentsData, error } = await supabase
        .from('Garment')
        .select('*')
        .eq('user_id', data.data.user.id);

      if (error) {
        console.error('Hubo un error al buscar las prendas:', error.message);
      } else {
          garmentsData.map((garment) => {
              const fileName = garment.image.split('/').pop();
               supabase.storage
                .from('images2')
                .download(fileName).then((response) => {
                  const imageUrl = URL.createObjectURL(response.data);
                  console.log('Imagen descargada:', imageUrl);
                    setGarments((prevState) => [...prevState, { ...garment, image: imageUrl }]);
        }).catch((error) => {
             console.error('Hubo un error al descargar la imagen:', error.message);
        });
            });

      }
    } catch (error) {
      console.error('Hubo un error al buscar las prendas:', error.message);
    }
  });
}, []);


  const handleUploadClick = () => {
    navigate('/uploadClothesForm');
  };

return (
  <div className="header">
    <h1>Closet</h1>
    <div className="button-container">
      <button onClick={handleUploadClick}>Upload Garment</button>
    </div>
    <div className="garment-container">
      {garments.map((garment) => (
        <div key={garment.id} className="garment-item">
          <img src={garment.image} alt={garment.type} width={200} />
        </div>
      ))}
    </div>
  </div>
);
}

export default Closet;
