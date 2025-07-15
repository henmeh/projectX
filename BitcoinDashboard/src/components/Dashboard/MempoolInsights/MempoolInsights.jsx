import CurrentMempoolVisualizer from "../CurrentMempoolVisualizer/CurrentMempoolVisualizer";
import FeeHistogram from "../FeeHistogram/FeeHistogram";


const MempoolInsights = () => {

    return (
        <div>
            <CurrentMempoolVisualizer />
            <FeeHistogram />
        </div>
    );
};

export default MempoolInsights;