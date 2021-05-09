# neuralTransfer
<h1>Neural Style Transfer</h1>
<p>Neural Style Transfer was first proposed in the paper "A Neural Algorithm of Artistic Style" (Gatys, et al., 2015). Neural style transfer uses the early layers of a trained convolutional neural network to represent the style of an image, as these layer capture low level features of the image. Layers deeper in the network capture the content. This script uses a pre-trained VGG19 model and extracts the following style layers from the style image: ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']. The content layer used was ['block3_conv2']. Results of applying this process using an image of the Louvre Museum for the content, and Salvador Dali's "The Persistence of Memory" for the style are shown below.</p>



<table>
  <tbody>
    <tr>
      <td>Style
      </td>
      <td>Content
      </td>
      <td>Result
      </td>
    </tr>
    <tr>
      <td>
        <img src="https://user-images.githubusercontent.com/17497237/117584950-d1db4000-b0dd-11eb-9cfb-cb7dc04c0cc1.jpg" width="250px">
      </td>
      <td>
        <img src="https://user-images.githubusercontent.com/17497237/117584945-ca1b9b80-b0dd-11eb-8e97-feabcbdc8742.jpg" width="250px">
      </td>
      <td>
        <img src="https://user-images.githubusercontent.com/17497237/117584926-b1ab8100-b0dd-11eb-9656-68fe8b8d6c11.png" width="250px">
      </td>
    </tr>
  </tbody>
</table>
  


