import java.util.Scanner;

public class Main {

    

    

    public static void solution3(){
        Scanner scanner = new Scanner(System.in);
        int t = scanner.nextInt();
        while (t-- > 0){
            int n;
            n = scanner.nextInt();
            int[] costs = new int[n];
            for (int i=0; i<n; i++) costs[i] = scanner.nextInt();
            //starts with Even index 0
            long minEven = costs[0];
            long minOdd = costs[1];
            long totalEven = minEven;
            long totalOdd = minOdd;
            long minCost = minEven * n + minOdd * n;
            for (int i=2; i<n; i++){
                if (i%2 == 1){
                    minOdd = Math.min(minOdd, costs[i]);
                    totalOdd += costs[i];
                }
                else{
                    minEven = Math.min(minEven, costs[i]);
                    totalEven += costs[i];
                }
                long this_cost = totalEven - minEven + minEven * (n - (i+2)/2 + 1) + totalOdd - minOdd + minOdd * (n - (i+1)/2 + 1);
                minCost = Math.min(minCost, this_cost);
            }
            System.out.println(minCost);
        }
        scanner.close();
    }

    

    public static void main(String[] args) {
        solution3();
    }
}
