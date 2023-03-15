import {ChangeEvent, useEffect, useState} from 'react';
// import Demo from './Demo.js';
import myHTML from "./Demo"
import conn from "./con.png"

// Component to upload files to the project
function FileUploadSingle() {
  const [file, setFile] = useState();
  const [response, setResponse] = useState([])


  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  };

  // useEffect(() => {
  //     handleUploadClick();
  // }, [])

  const handleUploadClick = () => {
    if (!file) {
      return;
    }

    // 👇 Uploading the file using the fetch API to the server
    fetch('http://127.0.0.1:5000/makePrediction', {
      method: 'POST',
      body: file,
      // 👇 Set headers manually for single file upload
      headers: {
        'content-type': file.type,
        'content-length': `${file.size}`, // 👈 Headers need to be a string
      },
    })
      .then((res) => res.json())
      .then((data) => {
        console.log(data);
        setResponse(data.toFixed(2));
      })
      .catch((err) => console.error(err));
  };

//   const myHTML = `<h1>John
// Doee</h1>`;

  return (
    <div>
        { response == 0 &&

        <>
        <label for="file"> Choose patient file   </label>
      <input id = 'file' type="file" onChange={handleFileChange} />

      <div>{file && `${file.name} - ${file.type}`}</div>
        <h3>   </h3>
      <button onClick={handleUploadClick}>Upload patient image for processing</button>
      <div>
         </div>
          </>
            }
          {/*{response() && response.length > 0 && response.map((responseObj, index) => (*/}
          {/*  <li key={responseObj}>{responseObj}</li>*/}
          {/*))}*/}
          {response > 0 &&
              <>
              <h3>Patient has probability {response} of being unhealthy</h3>
              <div dangerouslySetInnerHTML={{ __html: myHTML }}/>
                <img src={conn} alt="Connectivity Image" />
              </>
          }
          {/*{response.length > 0 => (*/}
          {/*    */}
          {/*    )*/}
          {/*}*/}


    </div>
  );
}

export default FileUploadSingle;