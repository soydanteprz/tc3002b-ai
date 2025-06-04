import java.util.*;
import java.io.*;
import java.lang.*;
import static java.lang.Math.*;

public class cp{
    static BufferedReader br;
    static StringTokenizer st;

    public static void main(String[] args)throws IOException{

        br = new BufferedReader(new InputStreamReader(System.in));
        List<Integer> answer = new ArrayList<Integer>();

        for(int t=Integer.parseInt(br.readLine()); t>0; t--){
            st = readLine();
            int n = tokenInt();

            String[] words = new String[n];
            int i=0;
            while(n-- >0){
                words[i] = br.readLine();
                i++;
            }

            answer.add(new Solver().solve(words));
        }

        for (int ans : answer ) System.out.println(ans);

    }

    static StringTokenizer readLine() throws IOException { return new StringTokenizer(br.readLine()); }
    static int tokenInt() { return Integer.parseInt(st.nextToken()); }
    
    
    
    

    
    

    
    

}


class Solver{
    public int solve(String[] words) throws IOException{

        char[] alphabets = {'a', 'b', 'c', 'd', 'e'};

        int[][] arr = new int[5][words.length];

        int i=0;
        for(String s : words) {
            char[] wordArr = s.toCharArray();
            int j=0;
            for (char alpha : alphabets) {
                int counter = 0;
                for(char letter : wordArr){
                    if(alpha == letter)
                        counter++;
                }
                arr[j][i] = 2*counter - s.length();
                j++;
            }
            i++;
        }

        int[] ans = new int[5];

        int m=0;
        for (int[] ar : arr) {
            Arrays.sort(ar);
            int val=0;
            int counter=0;
            for(i=ar.length-1; i>=0; i--){
                val += ar[i];
                if(val>0) counter++;
                else break;
            }
            ans[m] = counter;
            m++;
        }

        return Arrays.stream(ans).max().getAsInt();
    }
}
