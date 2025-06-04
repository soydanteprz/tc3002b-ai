import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.util.Stack;

// https://codeforces.com/contest/1407/problem/D

public class Discrete_Centrifugal_Jumps {
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        int n = scan.nextInt();
        int[] h = new int[n];
        for (int i = 0; i < n; i++) h[i] = scan.nextInt();
        System.out.println(getMinJump(h, n));
    }
    static int getMinJump(int[] h, int n) {
        int[] rightG = new int[n], leftG = new int[n];
        int[] rightS = new int[n], leftS = new int[n];
        Arrays.fill(rightG, -1); Arrays.fill(rightS, -1);
        Arrays.fill(leftG, -1); Arrays.fill(leftS, -1);
        Stack<Integer> stack = new Stack<>();

        // next greater in the right segment
        int i = 0;
        while (i < n) {
            if (!stack.empty() && h[i] >= h[stack.peek()]) rightG[stack.pop()] = i;
            else stack.push(i++);
        }
        stack = new Stack<>();
        // next smaller in the right segment
        i = 0;
        while (i < n) {
            if (!stack.empty() && h[i] <= h[stack.peek()]) rightS[stack.pop()] = i;
            else stack.push(i++);
        }
        stack = new Stack<>();
        // next greater in left segment
        i = n-1;
        while (i >= 0) {
            if (!stack.empty() && h[i] >= h[stack.peek()]) leftG[stack.pop()] = i;
            else stack.push(i--);
        }
        stack = new Stack<>();
        // next smaller in left segment
        i = n-1;
        while (i >= 0) {
            if (!stack.empty() && h[i] <= h[stack.peek()]) leftS[stack.pop()] = i;
            else stack.push(i--);
        }

        ArrayList<Integer>[] jump = new ArrayList[n];
        for (i = 0; i < n; i++) jump[i] = new ArrayList<>();
        for (i = 0; i < n; i++) {
            // max(h[i+1] ... h[j-1]) < min(h[i], h[j])
            if (rightG[i] != -1) jump[i].add(rightG[i]);
            if (leftG[i] != -1) jump[leftG[i]].add(i);

            // max(h[i], h[j]) < min(h[i+1] ... h[j])
            if (rightS[i] != -1) jump[i].add(rightS[i]);
            if (leftS[i] != -1) jump[leftS[i]].add(i);
        }
        int[] dp = new int[n]; Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for (int u = 0; u < n; u++) {
            for (int v: jump[u]) {
                dp[v] = Math.min(dp[v], dp[u] + 1);
            }
        }
        return dp[n-1];
    }
}
