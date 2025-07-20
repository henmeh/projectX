import WhaleTransactions from "../WhaleTransactions/WhaleTransactions"
import FeeHistogram from "../FeeHistogram/FeeHistogram";


const Mempool = () => {
  return (
    <>
        <WhaleTransactions/>
        <FeeHistogram/>
    </>
  );
};

export default Mempool;