import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

public class Jumps {

    public static void main(String[] args) {

        FastScanner sc = new FastScanner();
        int n = sc.nextInt();
        int ar[] = sc.nextIntArray(n);
        // 1 4 6 9 3 2
        int dp[] = new int[n];
        int nextLower[] = new int[n];
        int nextHigher[] = new int[n];
        int nextLowerE[] = new int[n];
        int nextHigherE[] = new int[n];
        Arrays.fill(nextHigher, n);
        Arrays.fill(nextLower, n);
        Arrays.fill(nextHigherE, n);
        Arrays.fill(nextLowerE, n);
        dp[n-1] = 0;
        Stack<Integer> stack = new Stack<>();

        for(int i = n-1; i >= 0; i--) {

            while(stack.size() > 0 && ar[stack.peek()] < ar[i])
                stack.pop();
            if(stack.size() > 0)
                nextHigherE[i] = stack.peek();
            stack.push(i);
        }
        stack.clear();

        for(int i = n-1; i >= 0; i--) {

            while(stack.size() > 0 && ar[stack.peek()] <= ar[i])
                stack.pop();
            if(stack.size() > 0)
                nextHigher[i] = stack.peek();
            stack.push(i);
        }
        stack.clear();

        for(int i = n-1; i >= 0; i--) {

            while(stack.size() > 0 && ar[stack.peek()] > ar[i])
                stack.pop();
            if(stack.size() > 0)
                nextLowerE[i] = stack.peek();
            stack.push(i);
        }
        stack.clear();

        for(int i = n-1; i >= 0; i--) {

            while(stack.size() > 0 && ar[stack.peek()] >= ar[i])
                stack.pop();
            if(stack.size() > 0)
                nextLower[i] = stack.peek();
            stack.push(i);
        }

        for(int i = n-2; i >= 0; i--) {

            int maxIndex = Math.max(nextHigherE[i], nextLowerE[i]);
            int ans = dp[i+1];
            if(maxIndex < n) {
                ans = Math.min(ans, dp[maxIndex]);
            }

            if(nextHigherE[i] < nextLowerE[i]) {
                int cur = nextHigherE[i];
                while(cur < maxIndex) {
                    ans = Math.min(ans, dp[cur]);
                    cur = nextLower[cur];
                }
            } else {
                int cur = nextLowerE[i];
                while(cur < maxIndex) {
                    ans = Math.min(ans, dp[cur]);
                    cur = nextHigher[cur];
                }
            }
            dp[i] = ans + 1;
        }
        System.out.println(dp[0]);
    }


    

    

    
    
    
    
    
    

    

    


    

    

    


    
}
