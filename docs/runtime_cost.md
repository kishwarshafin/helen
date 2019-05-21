<center>
<h1> Run-time case-study
</center>

To ensure robustness of `HELEN` we have tested it on two cloud computing platforms `Amazon Web Services (AWS)` and `Google Cloud Platform (GCP)`. We report runtime on multiple samples and the instance-types we have used while running the pipeline. The two computations we do with `HELEN` are:
* Run `call_consensus.py` (requires GPU).
* Run `stitch.py` (Multi-threaded on CPU, no GPU required).

#### Google Cloud Platform runtime
GCP allows to customize an instance between different runs. Users can stop an instance, scale it and start the next step. We ran `HELEN` on four samples in such way that is most cost-effective. We estimated the costs from the [Google Cloud Platform Pricing Calculator](https://cloud.google.com/products/calculator/).

<center>
<table>
  <tr>
    <th rowspan="2">Sample</th>
    <th colspan="4">call_consensus.py</th>
    <th colspan="3">stitch.py</sub></th>
    <th rowspan="2">Total<br>Cost</th>
  </tr>
  <tr>
    <th>Instance type</th>
    <th>GPU</th>
    <th>Time</th>
    <th>Cost</th>
    <th>Instance type</th>
    <th>Time</th>
    <th>Cost</th>
  </tr>
  <tr>
    <td>HG01109</td>
    <td>n1-standard-32</td>
    <td>2 x Tesla P100</td>
    <td>3:24:19</td>
    <td>19$</td>
    <td>n1-standard-32</td>
    <td>0:49:48</td>
    <td>3$</td>
    <td>22$</td>
  </tr>
  <tr>
    <td>GM24143</td>
    <td>n1-standard-32</td>
    <td>2 x Tesla P100</td>
    <td>3:46:19</td>
    <td>19$</td>
    <td>n1-standard-32</td>
    <td>1:02:27</td>
    <td>4$</td>
    <td>23$</td>
  </tr>
  <tr>
    <td>HG02080</td>
    <td>n1-standard-32</td>
    <td>2 x Tesla P100</td>
    <td>3:32:19</td>
    <td>19$</td>
    <td>n1-standard-96</td>
    <td>0:21:22</td>
    <td>2$</td>
    <td>21$</td>
  </tr>
  <tr>
    <td>HG01243</td>
    <td>n1-standard-32</td>
    <td>2 x Tesla P100</td>
    <td>3:53:19</td>
    <td>20$</td>
    <td>n1-standard-96</td>
    <td>0:20:21</td>
    <td>2$</td>
    <td>22$</td>
  </tr>
</table>
</center>

If you want to do all three steps without rescaling the instance after each step, we suggest you use this configuration:
* Instance type: n1-standard-32 (32 vCPUs, 120GB RAM)
* GPUs: 2 x NVIDIA Tesla P100
* Disk: 2TB SSD
* Cost: 4.65$/hour

The estimated runtime with this instance type is 6 hours. <br>
The estimated cost with this instance is <b>28$</b>.

#### Amazon Web Services
`AWS` does not provide an option to customize resources between steps. Hence, we used `p2.8xlarge` instance type for `HELEN`. The configuration is as follows:
* Instance type: p2.8xlarge (32 vCPUs, 488GB RAM)
* GPUs: 8 x NVIDIA Tesla K80
* Disk: 2TB SSD
* Cost: 7.20$/hour

The estimated runtime with this instance type: 5 hours <br>
The estimated cost in `AWS` is: <b>36$</b>
