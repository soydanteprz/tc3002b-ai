import java.util.*;
import java.io.*;

public class Main {

    public static void main(String[] args) throws IOException {
        Scanner sc = new Scanner(System.in);
        PrintWriter pw = new PrintWriter(System.out);
        int t = sc.nextInt();
        while(t-- > 0){
            int n = sc.nextInt();
            String[] s = new String[n];
            for(int i=0; i<n; i++)
                s[i] = sc.next();
            int MAX = 0;
            for(char c = 'a'; c <= 'e'; c++){
                PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder()); //Big comes in top;
                for(int i=0; i<n; ++i) {
                    int curChar = 0;
                    int otherChar = 0;
                    for(int j=0; j<s[i].length(); j++) {
                        if(s[i].charAt(j) == c)
                            curChar++;
                        else
                            otherChar++;
                    }
                    int diff = curChar - otherChar;
                    pq.add(diff);
                }
                int cur = 0;
                int numberOfWords = 0;
                while(!pq.isEmpty()){
                    if(cur + pq.peek() > 0){
                        cur += pq.poll();
                        numberOfWords++;
                    }else{
                        break;
                    }
                }
                MAX = Math.max(MAX, numberOfWords);
            }
            pw.println(MAX);
        }
        pw.close();
    }

    
}
