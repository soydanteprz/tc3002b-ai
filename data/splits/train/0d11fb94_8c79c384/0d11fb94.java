import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.*;


public class Main {
    static int i, j, k, n, m, t, y, x, sum = 0;
    static long mod = 1000000007;
    static FastScanner fs = new FastScanner();
    static PrintWriter out = new PrintWriter(System.out);
    static String str;
    static long ans;

    static List<Integer> zeros = new ArrayList<>();
    static List<Integer> ones = new ArrayList<>();
    static int oneCount = 0;
    static int[][] dp = new int[5005][5005];

    public static void main(String[] args) {
        t = 1;

        while (t-- > 0) {

            n = fs.nextInt();

            for(int i = 0;i<n;i++){
                x = fs.nextInt();
                if(x==1){
                    ones.add(i);
                    oneCount++;
                }
                else
                    zeros.add(i);
            }

            for(int i=0;i<n;i++){
                for(int j = 0; j<n;j++){
                    dp[i][j]=-1;
                }
            }

            out.println(minCost(0,0));
        }

        out.close();
    }

    static int minCost(int zIndex, int oIndex){

        if(oIndex == ones.size())
            return 0;
        if(zIndex == zeros.size())
            return 1000000007;

        if(dp[zIndex][oIndex]==-1)
            dp[zIndex][oIndex]= Math.min(Math.abs(zeros.get(zIndex) - ones.get(oIndex))+minCost(zIndex+1, oIndex+1) , minCost(zIndex+1, oIndex));

        return dp[zIndex][oIndex];
    }


    

    

    

    

    


}
