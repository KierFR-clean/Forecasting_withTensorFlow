import React, { useState, useCallback, useMemo } from "react";
import * as tf from "@tensorflow/tfjs";
import Papa from "papaparse";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

/**
 * @typedef {Object} SalesDataPoint
 * @property {string} 
 * @property {string} 
 * @property {number} 
 */

/**
 * @typedef {Object} ProcessedDataPoint
 * @property {number} 
 * @property {number} 
 * @property {number} 
 * @property {string} 
 * @property {string} 
 */

/**
 * @typedef {Object} ForecastDataPoint
 * @property {string} 
 * @property {string} 
 * @property {number} 
 * @property {number} 
 */

//set lang si model saka ung tracking niya
const SalesForecastApp = () => {
  const [salesData, setSalesData] = useState([]);
  const [forecastData, setForecastData] = useState([]);
  const [model, setModel] = useState(null);
  const [selectedProducts, setSelectedProducts] = useState([]);
  const [epochLosses, setEpochLosses] = useState([]);
  //usecallback para isahan lang na rendering tapos dictionaries para kay date at number
  // nagassigned ako ng unique numbers sa dalawa
  // kinuha ko ung early year sa sheet bale kinovert lang sa year then extract ung minimum
  //transform lang si array kapag previous date ung same id then kapag new increment lang same kay product
  //sa quantity same lang then inistore lang si orig date saka name pang reference later
  //scale lang sino min and max na quantities para inconvert siya numbers na interval sa 0, 1
  // apply normalization formula sa statistics like x - x.min /x.max - x.min 
  // example ung wilkins first quantity is 85 - lowest sa lahat ay 6 then divide kay 285 (ung highest) the kay 6 so 0.3 
  //then return ung normalized also scaler para sa trainmodel ung saka ung dictionaries 
  const preprocessData = useCallback((data) => {
    const dateMap = new Map();
    const productMap = new Map();
    let [dateCounter, productCounter] = [1, 0];

    const earliestYear = Math.min(
      ...data.map(({ date }) => new Date(date).getFullYear())
    );

    const processedData = data.map(({ date, product, quantity }) => ({
      date: dateMap.has(date)
        ? dateMap.get(date)
        : dateMap.set(date, dateCounter++).get(date),
      product: productMap.has(product)
        ? productMap.get(product)
        : productMap.set(product, productCounter++).get(product),
      quantity,
      originalDate: date,
      productName: product,
    }));

    const quantities = processedData.map(({ quantity }) => quantity);
    const scaler = {
      min: Math.min(...quantities),
      max: Math.max(...quantities),
    };

    const normalizedData = processedData.map((d) => ({
      ...d,
      quantity: (d.quantity - scaler.min) / (scaler.max - scaler.min),
    }));

    return {
      processedData: normalizedData,
      scaler,
      dateMap,
      productMap,
      earliestYear,
      uniqueProducts: [...new Set(data.map(({ product }) => product))],
    };
  }, []);
  //kunin ung nareturn 
  // gawa lang training data xs para sa input then ys para sa output then convert lang sa tensor2d para makuha ung trainingvalues saka shape
  // sa ai model ung first layer dalawang inputshape then 10 neurons processed the combined lang nangyari kay second layer 
  // nakaunit siya na 1 for isang quantity na prediction
  // set ung process sa adam pangadjust nung weights then measure lang ung loss 
  // nagadam ako incase may entries na meaningless sa csv ni sir
  // then train ung model by practising ng 100 times then callbacks lang para marecord sa console
  // once natapos ung process doon sa 32 batches then meaning non isang epoch natapos
  const trainModel = useCallback(async (data) => {
    const { processedData, scaler, dateMap, productMap, earliestYear } =
      preprocessData(data);

    const xs = tf.tensor2d(
      processedData.map(({ date, product }) => [date, product])
    );
    const ys = tf.tensor2d(
      processedData.map(({ quantity }) => [quantity])
    );

    const newModel = tf.sequential();
    newModel.add(
      tf.layers.dense({
        units: 10,
        activation: "relu",
        inputShape: [2]
      })
    );
    newModel.add(
      tf.layers.dense({ units: 1 })
    );

    newModel.compile({
      optimizer: "adam",
      loss: "meanSquaredError"
    });

    const losses = [];
    await newModel.fit(xs, ys, {
      epochs: 100,
      batchSize: 32,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss}`);
          losses.push({ epoch: epoch + 1, loss: logs.loss });
        },
      },
    });

    setEpochLosses(losses);
    setModel(newModel);
    generateForecast(newModel, processedData, scaler, dateMap, productMap, earliestYear);
  }, [preprocessData]);

  // kunin lang ung recent date then create nung reversed dict kasi id is pangpredict but name is pangdisplay
  // predict lang for six for 6 months the iconvert lang sa real numbers ung quantity kasi normalized siya 
  // sa actual scaled back lang sa orig range formula non is lq * (x.max - x.min) + x.min
  // then predicted ung converted na normalized number
  const generateForecast = useCallback((
    model,
    processedData,
    scaler,
    dateMap,
    productMap,
    earliestYear
  ) => {
    const uniqueProducts = [...new Set(processedData.map(({ product }) => product))];
    const maxDate = Math.max(...processedData.map(({ date }) => date));
    const reversedProductMap = new Map(
      [...productMap.entries()].map(([k, v]) => [v, k])
    );

    const forecasts = uniqueProducts.flatMap((product) => {
      const productData = processedData.filter(({ product: p }) => p === product);
      const lastActualQuantity = productData.at(-1).quantity;

      return Array.from({ length: 6 }, (_, i) => {
        const inputTensor = tf.tensor2d([[maxDate + i + 1, product]]);
        const prediction = model.predict(inputTensor);
        const predictedQuantity =
          prediction.dataSync()[0] * (scaler.max - scaler.min) + scaler.min;

        const monthNumber = ((maxDate + i) % 12) + 1;
        const yearOffset = Math.floor((maxDate + i - 1) / 12);
        const forecastYear = earliestYear + yearOffset;

        return {
          month: `${forecastYear}-${String(monthNumber).padStart(2, "0")}`,
          product: reversedProductMap.get(product),
          actual: Math.round(
            lastActualQuantity * (scaler.max - scaler.min) + scaler.min
          ),
          predicted: Math.round(predictedQuantity),
        };
      });
    });

    setForecastData(forecasts);
  }, []);


  // kada update nung page is nirerecall niya ulit function so i use useCallback para un na ulit gamitin niya incase may mabago doon sa depedencies
  // kunin yung file nakaoptional chaining siya kaya pag wala edi walang error also kapag isa lang siya so return nalang
  // papaparse para maparse siya = si headers kunin niya ung first row as header, skipemptylines para iignore niya lang empty line 
  //set si data na nkuna nung nagtapos ung parsing iprocess by unique tapos update lang si selected products then train na si model
  const handleFileUpload = useCallback(async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: async ({ data }) => {
        setSalesData(data);
        const { uniqueProducts } = preprocessData(data);
        setSelectedProducts(uniqueProducts);
        await trainModel(data);
      },
    });
  }, [preprocessData, trainModel]);

  const handleProductSelect = useCallback((event) => {
    const selectedValues = Array.from(event.target.selectedOptions, (opt) => opt.value);

    if (selectedValues.includes("All Products")) {
      setSelectedProducts([...new Set(forecastData.map(({ product }) => product))]);
    } else {
      setSelectedProducts(selectedValues);
    }
  }, [forecastData]);

  // instead na recalculate tong filtering i use useMemo para minamark niya na ung products
  //check lang kung ung product nandoon sa updated list para un lang kukunin
  const filteredForecastData = useMemo(
    () => forecastData.filter(({ product }) => selectedProducts.includes(product)),
    [forecastData, selectedProducts]
  );
  // pangdisplay using tailwind components 
  // si y-axis na auto scale based sa actual and predicted incase may lumampas si x-axix kasi needed for 6 months range lang
  return (
    <div className="min-h-screen min-w-screen flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="w-full max-w-7xl h-screen overflow-auto bg-slate-800 rounded-2xl shadow-2xl border border-gray-100">
        <header className="px-8 py-6 bg-gradient-to-r from-primary-500 to-primary-600 text-white">
          <h1 className="text-3xl font-extrabold text-center">
            My Sales Forecast Predictor
          </h1>
        </header>

        <main className="p-8 space-y-6">
          <div className="flex flex-wrap gap-4 items-center justify-between">
            <input
              type="file"
              accept=".txt,.csv"
              onChange={handleFileUpload}
              className="cursor-pointer block w-full text-sm file:py-2 file:px-4 
                file:mr-4 file:rounded-full file:border-0 file:font-semibold 
                file:bg-primary-50 file:text-primary-600 hover:file:bg-primary-100 
                focus:ring-2 focus:ring-primary-500 transition"
            />
            {forecastData.length > 0 && (
              <select
                value={selectedProducts}
                onChange={handleProductSelect}
                className="px-8 text-center border rounded-lg p-2 focus:ring-2 focus:ring-primary-900 cursor-pointer"
              >
                <option value="All Products">All Products</option>
                {[...new Set(forecastData.map(({ product }) => product))].map((product) => (
                  <option key={product} value={product}>
                    {product}
                  </option>
                ))}
              </select>

            )}
          </div>

          {filteredForecastData.length > 0 && (
            <>
              <ResponsiveContainer width="100%" height={500}>
                <LineChart data={filteredForecastData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    stroke="#C21807"
                    name="Actual Sales"
                  />
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="#F4C430"
                    name="Predicted Sales"
                  />
                </LineChart>
              </ResponsiveContainer>

              <ForecastTable data={filteredForecastData} />
            </>
          )}
        </main>
      </div>
    </div>
  );
};

/**
 * @param {{ data: ForecastDataPoint[] }} 
 */
// report of forecast 
const ForecastTable = ({ data }) => (
  <>
    <h2 className="text-2xl font-bold mb-4 flex justify-center">
      Forecast Data
    </h2>
    <table className="table-auto w-full text-center border-collapse shadow-lg rounded-lg overflow-hidden">
      <thead className="bg-gradient-to-r from-blue-500 to-indigo-500 text-white">
        <tr>
          <th className="p-4 text-base font-semibold tracking-wide border-b border-indigo-300">
            Month
          </th>
          <th className="p-4 text-base font-semibold tracking-wide border-b border-indigo-300">
            Product
          </th>
          <th className="p-4 text-base font-semibold tracking-wide border-b border-indigo-300">
            Actual Sales
          </th>
          <th className="p-4 text-base font-semibold tracking-wide border-b border-indigo-300">
            Predicted Sales
          </th>
        </tr>
      </thead>
      <tbody>
        {data.map(({ month, product, actual, predicted }, idx) => (
          <tr
            key={`${month}-${product}`}
            className={`${idx % 2 === 0 ? "bg-gray-50" : "bg-gray-100"
              } hover:bg-indigo-100 transition-all`}
          >
            <td className="p-3 text-base text-gray-800 border-b border-gray-200">
              {month}
            </td>
            <td className="p-3 text-base text-gray-800 border-b border-gray-200">
              {product}
            </td>
            <td className="p-3 text-base text-gray-800 border-b border-gray-200">
              {actual}
            </td>
            <td className="p-3 text-base text-gray-800 border-b border-gray-200">
              {predicted}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  </>
);

export default SalesForecastApp;