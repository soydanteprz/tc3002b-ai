import java.io.*;
import java.util.*;
public class Solution{
    public static int in = 0, count=0;
    static class comparator implements Comparator<int[]>{
        
    }
    public static int solve(ArrayList<int[]> al, int[] total){
        ArrayList<int[]> c = (ArrayList<int[]>) al.clone();
        java.util.Collections.sort(c, new comparator());

        int i= al.size()-1;
        int sum=0; for(int ii:total) sum+=ii;
        int check = total[in];
        sum-=check;
        for(; i>=0; i--){
            if(check>sum) return i+1;
            else{
                int newSum=0; for(int ii:c.get(i)) newSum+=ii;
                newSum-=c.get(i)[in];
                sum-=newSum;
                check-=c.get(i)[in];
            }
        }
        return 0;
    }
    public static void main (String[] args) throws java.lang.Exception {
        FastReader sc = new FastReader();
        BufferedWriter out = new BufferedWriter(new OutputStreamWriter(System.out));
        int testCase = sc.nextInt();
        while (testCase-->0){
            int n = sc.nextInt();
            String[] strArr = new String[n];
            for(int i=0; i<n; i++) {
                strArr[i]=sc.nextLine();
            }
            int[] total = new int[5];
            ArrayList<int[]> al = new ArrayList<>();
            for(int i=0; i<n; i++){
                int[] arr= new int[5];
                for(int j=0; j<strArr[i].length(); j++){
                    arr[strArr[i].charAt(j)-'a']++;
                }
                for(int j=0; j<5; j++){
                    total[j]+=arr[j];
                }
                al.add(arr);
            }
            int ans = Integer.MIN_VALUE;
            for(int i=0; i<5; i++) {
                in = i;
                ans = Math.max(solve(al, total), ans);
                count=0;
            }
            System.out.println(ans);
        }
    }
    // Fast Reader Class
    
}
