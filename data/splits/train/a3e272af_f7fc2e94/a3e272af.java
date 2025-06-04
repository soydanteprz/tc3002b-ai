import java.io.*;
import java.util.*;

public class Asd {

    static PrintWriter w = new PrintWriter(System.out);
    static FastScanner s = new FastScanner();
    static boolean sd = false;
  
    public static void main(String[] args) {

       int t = s.nextInt();
        //int t=1;
        while (t-- > 0) {
            solve();
        }
        w.close();

    }
  public static class Student {
    public int i1;
    public int value;
          
    // A parameterized student constructor
    public Student(int i1,int i2) {
      
        this.i1 = i1;
        this.value=i2;
    }
      
    public int getkey() {
        return i1;
    } 
    public int getValue() {
        return value;
    } 
}
   static class StudentComparator implements Comparator<Student>{
              
            // Overriding compare()method of Comparator 
                        // for descending order of cgpa
        
        }
    
    /* Function to print all the permutations of the string
     static String swap(String str, int i, int j)
     {
     char ch;
     char[] array = str.toCharArray();
     ch = array[i];
     array[i] = array[j];
     array[j] = ch;
     return String.valueOf(array);
     }
    
     static void permute(String str,int low,int high)
     {
     if(low == high)
     list.add(Long.parseLong(str));
 
     int i;
     for(i = low; i<=high; i++){
     str = swap(str,low,i);
     permute(str, low+1,high);
     str = swap(str,low,i);
     }
     }
     use  permute(str2,0,str2.length()-1); to perform combinations
     */

    public static void solve() {
        int n=s.nextInt();
        int m=s.nextInt();
        int x=s.nextInt();
        int arr[]=new int[n];int res[]=new int[n];
        for(int i=0;i<n;i++)
            arr[i]=s.nextInt();
        
        PriorityQueue<Student> pq=new PriorityQueue<Student>(new StudentComparator());
        for(int i=0;i<m;i++){
            pq.add(new Student(arr[i],i));res[i]=i;}
        for(int i=m;i<n;i++)
        {
            Student s1=pq.poll();
            int k2=s1.getkey()+arr[i];
            int v2=s1.getValue();res[i]=v2;
            pq.add(new Student(k2,v2));
        }
        w.println("YES");
        for(int i=0;i<n;i++)
            w.print(res[i]+1+" ");
        w.println();
       
    }

    

    

    

    

    

    

}
