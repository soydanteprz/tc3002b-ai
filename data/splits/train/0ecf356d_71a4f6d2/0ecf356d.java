import java.io.*;
import java.util.*;

public class Main {

    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int n = Integer.parseInt(br.readLine());
        StringTokenizer st = new StringTokenizer(br.readLine());
        int[] h = new int[n];
        int[] dp = new int[n];
        for(int i = 0; i<n; i++){
            h[i] = Integer.parseInt(st.nextToken());
        }
        Stack<Integer> hi = new Stack<>();
        Stack<Integer> lo = new Stack<>();
        hi.push(0);
        lo.push(0);
        for(int i = 1; i<n; i++){
            dp[i] = dp[i-1]+1;
            while(!hi.isEmpty() && h[hi.peek()]<h[i]){
                dp[i] = Math.min(dp[i], dp[hi.peek()]+1);
                hi.pop();
            }
            if(!hi.isEmpty()){
                dp[i] = Math.min(dp[i], dp[hi.peek()]+1);
                if(h[i] == h[hi.peek()]) hi.pop();
            }
            while(!lo.isEmpty() && h[lo.peek()]>h[i]){
                dp[i] = Math.min(dp[i], dp[lo.peek()]+1);
                lo.pop();
            }
            if(!lo.isEmpty()){
                dp[i] = Math.min(dp[i], dp[lo.peek()]+1);
                if(h[i] == h[lo.peek()]) lo.pop();
            }
            hi.push(i);
            lo.push(i);
        }
        System.out.println(dp[n-1]);
    }
}
