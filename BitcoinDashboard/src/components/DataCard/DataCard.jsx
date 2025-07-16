import React from 'react';
import { Card, Typography } from 'antd';

const { Title, Text } = Typography;

/**
 * DataCard component to display various types of data within a consistent card layout.
 *
 * @param {string} className - CSS class name for custom styling.
 * @param {string} title - Title of the card.
 * @param {string|JSX.Element} data - Data to display; can be text or HTML.
 * @param {string} key - Unique key for the component.}
*/
const DataCard = ({ className, key, title, data }) => {
  return (
    <Card className={className} style={{height: "100%"}} key={key} title={<Title level={5} style={{ margin: 0 }}>{title}</Title>}>
      {typeof data === 'string' ? (
        <Text>{data}</Text>
      ) : (
        <div>{data}</div>
      )}
    </Card>
  );
};

export default DataCard;