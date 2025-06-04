import java.io.BufferedReader;
import java.io.IOException;
import java.io.*;
import java.util.*;

import javax.management.Query;
public class Test{
    public static void main(String[] args) throws IOException, InterruptedException{
      Scanner sc = new Scanner(System.in);
     int t = sc.nextInt();
    
     while(t-->0){
      int n = sc.nextInt();
      String [] words = new String[n];
     
      for(int i =0;i<n;i++){
        words[i] = sc.nextLine();
       
      }
      int maxRes =0;
      for(int i =0;i<5;i++){
      int maxChar = 'a' +i;
      
      PriorityQueue<Pair> pq = new PriorityQueue<>();
      for (String word : words){
          pq.add(new Pair(word,occOfMaxChar(word, maxChar)-occOfOtherChar(word, maxChar)));
      }
      int res = 0;
      int curr = 0;
      int maxCharCount = 0;
      int otherCharCount =0;
      while(!pq.isEmpty()){
        String word = pq.poll().x;
        maxCharCount +=occOfMaxChar(word, maxChar);
         otherCharCount += occOfOtherChar(word, maxChar);
         curr ++;
         if(maxCharCount >otherCharCount){
           res = curr;
         }
          
      }
      maxRes = Math.max(maxRes, res);
    }
    System.out.println(maxRes);}
   
  }
  public static int occOfMaxChar (String s, int maxChar){
     int occ = 0;
     for(int i =0 ;i<s.length();i++){
       if(s.charAt(i)==maxChar){
         occ++;
       }
      }
      return occ;
  }
  public static int occOfOtherChar (String s, int maxChar){
    int occ = 0;
    for(int i =0 ;i<s.length();i++){
      if(s.charAt(i)!=maxChar){
        occ++;
      }
     }
     return occ;
 }

    static int w;
    static int n;
    static long [][] memo;
    static int [] depth ;
    static long[] values;
    static ArrayList<Pair> gold ;
    


    static class Pair implements Comparable {
      String x;
      int y;
      public Pair (String x, int y) {
        this.x = x;
        this.y = y;
      }
      
    }
    
     }
