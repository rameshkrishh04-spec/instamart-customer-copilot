-- Q1: WHERE then HAVING — completed orders above INR 500, customer total above INR 8,000.
SELECT customer_id, ROUND(SUM(order_amount), 2) AS qualifying_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY qualifying_spend DESC;

-- Q2: Three-table join — revenue by city and product category.
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS total_revenue
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products p ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3: Scalar subquery — products priced above the average catalogue price.
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: LEFT JOIN — customers with no completed order.
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: CTEs — customers whose completed spend is above the average customer spend.
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders WHERE status = 'completed'
    GROUP BY customer_id
), average_spend AS (
    SELECT AVG(total_spend) AS avg_spend FROM customer_spend
)
SELECT cs.customer_id, ROUND(cs.total_spend, 2) AS total_spend,
       ROUND(a.avg_spend, 2) AS average_spend
FROM customer_spend cs CROSS JOIN average_spend a
WHERE cs.total_spend > a.avg_spend
ORDER BY total_spend DESC;

-- Q6: Ranking window functions by city.
WITH customer_spend AS (
    SELECT c.city, o.customer_id, SUM(o.order_amount) AS total_spend
    FROM orders o JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, o.customer_id
)
SELECT city, customer_id, ROUND(total_spend, 2) AS total_spend,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rank_value,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rank_value,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC, customer_id) AS row_number_value
FROM customer_spend
ORDER BY city, rank_value, customer_id;

-- Q7: Daily completed revenue and cumulative revenue.
WITH daily AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day, SUM(order_amount) AS daily_revenue
    FROM orders WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT day, ROUND(daily_revenue, 2) AS daily_revenue,
       ROUND(SUM(daily_revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily ORDER BY day;